"""The evaluation matrix: every probe, in every memory condition, answered and judged.

Design
------
The audit compares *conditions*, not models. For each probe we ask the same answerer
the same question twice:

  full        the whole rendered transcript (baseline — what the agent would have had)
  compressed  the output of each compression strategy, one condition per strategy

so a difference in accuracy is attributable to the memory, not to the question or the
model. Each answer is then graded by the independent judge (``judge.py``) and the exact
context size is recorded with ``count_tokens`` of the string that was actually sent.

Why the run has to be resumable
-------------------------------
A full run is ``n_sessions * n_probes * (1 + n_strategies)`` answer calls plus the same
number of judge calls — several hundred round-trips through a free gateway that fails
~10-20% of requests and can take 30s each. Losing all of it to one crash is unacceptable,
so ``answers.jsonl`` is appended **and flushed per row**: the file on disk is always a
valid prefix of the run. On start, completed cells are loaded and skipped.

Concurrency
-----------
``cfg.concurrency`` workers (default 4) run the blocking HTTP calls in a
``ThreadPoolExecutor``. The only shared mutable state is the output file, and every
append happens under a single lock, so rows never interleave. Ordering in the file is
therefore completion order, not submission order — which is why every consumer
(``cost``, ``failures``) sorts or keys by ``(probe_id, strategy, condition)`` instead of
trusting file order.

Attribution caveat
------------------
``Answer.prompt_tokens`` / ``completion_tokens`` are computed with our tokenizer from the
exact prompt and reply. The shared ``UsageLedger`` cannot attribute records to the thread
that caused them, so a per-cell delta would be wrong under concurrency; the authoritative
model-reported totals live in ``usage.json``. The headline metric never uses these fields —
it uses ``context_tokens`` (tokenizer, deterministic) and ``n_correct``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from memfaith.schema import Answer, CompressionResult, Probe, Session
from memfaith.tokens import count_message_tokens, count_tokens

from .judge import judge_verdict, verdict_label
from . import (
    COMPRESSED_CONDITION,
    FULL_CONDITION,
    FULL_STRATEGY,
    AnswerRow,
    answer_key,
    as_dict,
    load_answers,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWERER_SYSTEM = """\
You are an AI agent continuing work on a long-running project. Below is your MEMORY of \
a past session: {memory_kind}. Answer the user's question using only that memory.

Rules:
- Answer with the specific value(s) asked for, in at most one or two short sentences.
- Do not explain your reasoning and do not restate the question.
- If the memory does not contain the answer, reply exactly: I don't know."""

#: ``{memory_kind}`` is filled with this so the answerer knows what it is reading.
MEMORY_KIND_FULL = "the complete transcript of the session"
MEMORY_KIND_COMPRESSED = "a compressed summary of the session"

ANSWERER_MAX_TOKENS = 400


def build_answerer_messages(context: str, question: str, *, compressed: bool) -> list[dict]:
    """Build the answerer prompt: role framing, then memory, then the question.

    One user message rather than a system+user pair: some free OpenAI-compatible
    gateways reject or silently drop a ``system`` role, and the framing works just as
    well inline. The memory is placed *before* the question so the question is the last
    thing the model reads — the ordering that gets the best instruction-following out of
    small models.
    """
    system = ANSWERER_SYSTEM.format(
        memory_kind=MEMORY_KIND_COMPRESSED if compressed else MEMORY_KIND_FULL
    )
    body = (
        f"{system}\n\n"
        f"MEMORY\n{context}\n\n"
        f"QUESTION\n{question.strip()}\n\n"
        f"ANSWER:"
    )
    return [{"role": "user", "content": body}]


# ---------------------------------------------------------------------------
# Rendering the full context
# ---------------------------------------------------------------------------
def render_full_context(session: Session) -> str:
    """Render the full transcript exactly as the dataset module does.

    ``CONTRACT.md`` makes ``memfaith.dataset.generator.render_transcript`` the single
    canonical renderer so the "full" condition and the compressors' input are the same
    bytes. That module is written in parallel with this one, so the import is lazy and
    falls back to the identical format rather than failing at import time.
    """
    try:
        from memfaith.dataset.generator import render_transcript as _canonical

        return _canonical(session)
    except Exception:  # noqa: BLE001 - dataset module may not exist yet
        return "".join(f"{t.role.upper()}: {t.text}\n" for t in session.turns)


# ---------------------------------------------------------------------------
# Compression lookup
# ---------------------------------------------------------------------------
def _group_compressions(
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult],
) -> dict[str, dict[str, CompressionResult]]:
    """Index compressions as ``{strategy: {session_id: result}}``.

    The contract calls the parameter ``compressors`` and ``run.py`` passes a dict keyed
    by strategy; both shapes are accepted so the harness cannot be broken by a caller
    that hands over the flat list from ``compressions.jsonl``.
    """
    grouped: dict[str, dict[str, CompressionResult]] = {}
    if isinstance(compressions, dict):
        items: Iterable[Any] = (
            r for rows in compressions.values() for r in (rows or [])
        )
    else:
        items = compressions
    for raw in items:
        row = as_dict(raw)
        strategy = str(row.get("strategy") or "")
        session_id = str(row.get("session_id") or "")
        if not strategy or not session_id:
            continue
        grouped.setdefault(strategy, {})[session_id] = (
            raw if isinstance(raw, CompressionResult) else CompressionResult(**row)
        )
    return grouped


def _strategy_order(
    grouped: dict[str, dict[str, CompressionResult]], cfg: Any
) -> list[str]:
    """Which strategies to evaluate, in order.

    ``cfg.strategies`` is authoritative: those are the arms the run is *supposed* to
    have. A configured arm with no compression rows yields ``compression_missing`` cells,
    which is the honest outcome — silently skipping it would make a half-run look like a
    two-arm study. Any extra strategy found in the data (e.g. an ad-hoc third arm) is
    evaluated after the configured ones rather than dropped.
    """
    configured = list(getattr(cfg, "strategies", ()) or ())
    return configured + [s for s in grouped if s not in configured]


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------
#: Errors that will not change if we retry: the compression itself is broken, so
#: re-answering an empty/garbage memory would only burn quota.
_PERMANENT_ERRORS = ("compression_missing", "compression_failed", "session_missing")


def _is_complete(row: AnswerRow) -> bool:
    """True when a persisted row needs no further work.

    Complete means *judged*: no error and a non-``None`` verdict. A row whose judge
    call failed is deliberately **not** complete — the retry is the whole point of the
    resume logic. Permanent structural errors are complete because retrying cannot help.
    """
    error = str(row.get("error") or "")
    if error.startswith(_PERMANENT_ERRORS):
        return True
    return not error and row.get("correct") is not None


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
def _answer_cell(
    *,
    probe: Probe,
    strategy: str,
    condition: str,
    context: str,
    client: Any,
) -> Answer:
    """Answer one probe under one condition, then judge it.

    Returns an ``Answer`` in every case. Gateway failures are recorded on the row
    (``error``) rather than raised: the cell will be retried on the next run, and the
    metrics exclude errored rows from the accuracy denominator so an infrastructure
    failure can never masquerade as a compression failure.
    """
    compressed = condition == COMPRESSED_CONDITION
    messages = build_answerer_messages(context, probe.question, compressed=compressed)
    row = Answer(
        probe_id=probe.probe_id,
        session_id=probe.session_id,
        strategy=strategy,  # type: ignore[arg-type]
        condition=condition,  # type: ignore[arg-type]
        answer_text="",
        context_tokens=count_tokens(context),
        # Tokenizer-derived per-cell attribution; see the module docstring.
        prompt_tokens=count_message_tokens(messages),
    )

    t0 = time.time()
    try:
        text = client.chat(
            "answerer",
            messages,
            temperature=0.0,
            max_tokens=ANSWERER_MAX_TOKENS,
            tag=f"answer:{probe.probe_id}:{strategy}",
        )
    except Exception as exc:  # noqa: BLE001 - one cell must not abort the run
        row.error = f"answer_error: {type(exc).__name__}: {exc}"
        row.latency_s = round(time.time() - t0, 2)
        return row
    row.answer_text = (text or "").strip()
    row.completion_tokens = count_tokens(row.answer_text)

    try:
        verdict = judge_verdict(client, probe, row.answer_text)
        row.correct = verdict.correct
        row.judge_raw = verdict.raw
        row.judge_confidence = verdict.confidence
    except Exception as exc:  # noqa: BLE001
        # The answer exists but could not be graded: keep it, mark it, retry later.
        row.error = f"judge_error: {type(exc).__name__}: {exc}"
    row.latency_s = round(time.time() - t0, 2)
    return row


def _append_row(path: str, row: Answer, lock: threading.Lock) -> None:
    """Append one row to ``answers.jsonl`` and flush it to disk, under a lock.

    ``flush()`` (not just ``write``) is what makes the file a real crash boundary: the
    row is on disk before the next cell starts, so a hard kill costs at most the cell
    in flight.
    """
    line = json.dumps(row.to_dict(), ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def _build_cells(
    sessions: list[Session],
    probes: list[Probe],
    grouped: dict[str, dict[str, CompressionResult]],
    strategies: list[str],
) -> list[dict[str, Any]]:
    """Enumerate every ``(probe, strategy, condition, context)`` cell to evaluate.

    Cells whose memory cannot exist (unknown session, missing/failed compression) are
    returned too, flagged with a permanent error, so they are persisted once and then
    skipped by the resume logic instead of being silently forgotten.
    """
    sessions_by_id = {s.session_id: s for s in sessions}
    cells: list[dict[str, Any]] = []
    for probe in probes:
        session = sessions_by_id.get(probe.session_id)
        if session is None:
            cells.append({"probe": probe, "strategy": FULL_STRATEGY,
                          "condition": FULL_CONDITION, "context": None,
                          "error": "session_missing"})
            continue
        full_text = render_full_context(session)
        cells.append({"probe": probe, "strategy": FULL_STRATEGY,
                      "condition": FULL_CONDITION, "context": full_text, "error": ""})
        for strategy in strategies:
            comp = grouped.get(strategy, {}).get(session.session_id)
            if comp is None:
                cells.append({"probe": probe, "strategy": strategy,
                              "condition": COMPRESSED_CONDITION, "context": None,
                              "error": "compression_missing"})
            elif comp.meta.get("error"):
                cells.append({"probe": probe, "strategy": strategy,
                              "condition": COMPRESSED_CONDITION, "context": None,
                              "error": f"compression_failed: {comp.meta['error']}"})
            else:
                cells.append({"probe": probe, "strategy": strategy,
                              "condition": COMPRESSED_CONDITION,
                              "context": comp.text, "error": ""})
    return cells


def _error_row(cell: dict[str, Any]) -> Answer:
    """Materialise a permanently-failed cell as an ``Answer`` row."""
    probe: Probe = cell["probe"]
    return Answer(
        probe_id=probe.probe_id,
        session_id=probe.session_id,
        strategy=cell["strategy"],  # type: ignore[arg-type]
        condition=cell["condition"],  # type: ignore[arg-type]
        answer_text="",
        correct=None,
        error=cell["error"],
    )


def _sort_key(row: AnswerRow) -> tuple[str, str, int, str]:
    """Deterministic output order: session, probe, baseline-first, strategy.

    Sorting on the row's own ids (rather than trusting file order, which is completion
    order under concurrency) makes the returned list — and therefore every downstream
    metric — independent of how the thread pool happened to interleave. The baseline
    sorts first within a probe, then strategies alphabetically, so no module-level state
    is needed and two concurrent runs cannot interfere.
    """
    strategy = str(row.get("strategy") or "")
    return (
        str(row.get("session_id") or ""),
        str(row.get("probe_id") or ""),
        0 if strategy == FULL_STRATEGY else 1,
        strategy,
    )


def run_matrix(
    sessions: list[Session],
    probes: list[Probe],
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult],
    client: Any,
    cfg: Any,
    out_dir: str,
) -> list[AnswerRow]:
    """Run the probe x condition matrix, resuming from ``out_dir/answers.jsonl``.

    Parameters
    ----------
    sessions, probes
        Dataset and probe objects (this module deliberately does not import the
        generators — they are owned by another module and must not become a runtime
        dependency of the metrics layer).
    compressions
        ``{strategy: [CompressionResult]}`` as produced by ``run.py`` (a flat iterable
        is also accepted).
    client
        ``LLMClient``-compatible object exposing ``chat(role, messages, ...)``.
    cfg
        ``RunConfig``; only ``strategies`` and ``concurrency`` are read.
    out_dir
        Directory for ``answers.jsonl`` (and the run summary).

    Returns
    -------
    list[dict]
        Every answer row for the run — resumed and newly produced — deduplicated and in
        deterministic order. Rows are exactly ``Answer.to_dict()`` so ``run.py`` can
        reload them with ``read_jsonl``. A summary of what this invocation did is
        written to ``out_dir/eval_summary.json``.
    """
    os.makedirs(out_dir, exist_ok=True)
    answers_path = os.path.join(out_dir, "answers.jsonl")

    grouped = _group_compressions(compressions)
    strategies = _strategy_order(grouped, cfg)

    cells = _build_cells(sessions, probes, grouped, strategies)
    existing = load_answers(answers_path)
    done = {answer_key(r) for r in existing if _is_complete(r)}

    pending = [
        c for c in cells
        if (c["probe"].probe_id, c["strategy"], c["condition"]) not in done
    ]

    # Permanent failures are written once and never retried.
    permanent = [c for c in pending if c["error"]]
    runnable = [c for c in pending if not c["error"]]

    lock = threading.Lock()
    produced: list[AnswerRow] = []
    n_errors = 0
    for cell in permanent:
        row = _error_row(cell)
        _append_row(answers_path, row, lock)
        produced.append(row.to_dict())
        n_errors += 1

    if runnable:
        concurrency = max(1, int(getattr(cfg, "concurrency", 4) or 1))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures: dict[Future[Answer], dict[str, Any]] = {
                pool.submit(
                    _answer_cell,
                    probe=cell["probe"],
                    strategy=cell["strategy"],
                    condition=cell["condition"],
                    context=cell["context"],
                    client=client,
                ): cell
                for cell in runnable
            }
            for fut in as_completed(futures):
                cell = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 - defensive: worker should not raise
                    probe = cell["probe"]
                    row = Answer(
                        probe_id=probe.probe_id,
                        session_id=probe.session_id,
                        strategy=cell["strategy"],  # type: ignore[arg-type]
                        condition=cell["condition"],  # type: ignore[arg-type]
                        answer_text="",
                        error=f"worker_error: {type(exc).__name__}: {exc}",
                    )
                if row.error:
                    n_errors += 1
                # Persist first, then keep in memory: a crash after this line loses nothing.
                _append_row(answers_path, row, lock)
                produced.append(row.to_dict())

    # The file is authoritative; reload so the returned set includes resumed rows too.
    merged = {answer_key(r): r for r in existing}
    for row in produced:
        merged[answer_key(row)] = row
    final_rows = sorted(merged.values(), key=_sort_key)

    summary = {
        "n_cells": len(cells),
        "n_resumed": len(cells) - len(pending),
        "n_new": len(produced),
        "n_errors": n_errors,
        "strategies": strategies,
        "answers_path": answers_path,
        "by_arm": _arm_counts(final_rows),
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return final_rows


def _arm_counts(rows: list[AnswerRow]) -> dict[str, dict[str, int]]:
    """Quick per-arm counts for the run log (the real metrics are in ``cost.py``).

    ``n_partial`` is derived with :func:`memfaith.metrics.judge.verdict_label`, the same
    function ``cost.py`` uses, so the log and ``metrics.json`` can never disagree about
    what counts as a near-miss.
    """
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        arm = str(row.get("strategy") or FULL_STRATEGY)
        slot = out.setdefault(arm, {"n": 0, "n_correct": 0, "n_partial": 0,
                                    "n_unjudged": 0, "n_errors": 0})
        slot["n"] += 1
        if row.get("error"):
            slot["n_errors"] += 1
        label = verdict_label(str(row.get("judge_raw") or ""), row.get("correct"))
        if row.get("correct") is True:
            slot["n_correct"] += 1
        elif label == "unparsed":
            slot["n_unjudged"] += 1
        elif label == "partial":
            slot["n_partial"] += 1
    return out
