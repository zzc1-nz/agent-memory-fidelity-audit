"""Failure analysis: which facts compression dropped, and what predicts the loss.

The question this answers
-------------------------
``cost.py`` says *whether* compression lost accuracy. This module says *what* it lost
and *why*, by joining every probe that flipped correct -> incorrect back to the
ground-truth facts it depends on. Those facts carry the covariates the dataset builder
planted (frequency, salience, embedding depth, stability), so the flip rate can be
decomposed into statements like:

    "facts mentioned once survived 41% of the time; facts mentioned 3+ times survived 88%"

Two different notions of "survival", kept deliberately separate
---------------------------------------------------------------
1. **Verbatim survival** (this module's ``survival_by_attribute``): the fact's canonical
   ``value`` string still appears as a substring of the compressed text. This is a
   *mechanical* check — no model involved — so it is reproducible and cheap. It can be
   pessimistic (a summary may convey "the key lives in `.env.production`" as
   "API key: .env.production", which still contains the value) but it never credits a
   summary that dropped the literal string the task needs.

2. **Answer survival** (``cost.py`` accuracy): the agent, reading the compressed text,
   still answers the probe correctly. This is what matters commercially, but it is
   noisy (LLM answerer + LLM judge) and confounded by the model's ability to guess.

Reporting both is the point: a fact can survive verbatim and still not be used
(``case.answer_text`` wrong while ``survived_verbatim`` is true — the model had the
value and failed to retrieve it), or be paraphrased away and still be answered
correctly from surrounding context.

Precise definition of ``survival_rate`` (quoted in the report)
--------------------------------------------------------------
For one attribute group (e.g. ``embedding``) and one attribute value (e.g.
``deep_buried``)::

    survival_rate = (# distinct critical facts with that attribute value that
                     (a) are referenced by at least one probe evaluated in both
                     conditions, and
                     (b) whose canonical value still appears verbatim — case- and
                     whitespace-insensitive substring match — in at least one of the
                     compressed memories those probes were given)
                    / (# distinct critical facts with that attribute value that
                       satisfy (a))

So the denominator is *probed* critical facts, not all facts: a fact no probe asks about
cannot be scored. A fact counts as surviving if it survived in **any** compressed arm
the probes were evaluated against; per-strategy numbers are also emitted under
``survival_by_attribute_by_strategy`` so the arms are not conflated.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from memfaith.schema import Answer, CompressionResult, Probe, Session

from . import (
    FULL_STRATEGY,
    AnswerRow,
    as_dict,
    dedupe_answers,
)

#: Frequency buckets from the README's hypothesis table (1 / 2 / 3+ mentions).
FREQUENCY_BUCKETS = ("1", "2", "3+")
ATTRIBUTE_GROUPS = ("embedding", "salience", "frequency", "stability")
_ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "embedding": ("standalone", "embedded", "deep_buried"),
    "salience": ("foreground", "background"),
    "stability": ("stable", "superseded"),
    "frequency": FREQUENCY_BUCKETS,
}

_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Lowercase + collapse whitespace, for a forgiving-but-literal substring check."""
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def frequency_bucket(frequency: int) -> str:
    """Map a mention count onto the ``1`` / ``2`` / ``3+`` buckets."""
    if frequency <= 1:
        return "1"
    if frequency == 2:
        return "2"
    return "3+"


def _fact_attributes(fact: dict[str, Any]) -> dict[str, str]:
    """The four covariates for one fact, with ``frequency`` bucketed."""
    mentions = fact.get("mentions") or []
    freq = fact.get("frequency")
    if freq is None:
        freq = len(mentions) if mentions else 0
    return {
        "embedding": str(fact.get("embedding") or "unknown"),
        "salience": str(fact.get("salience") or "unknown"),
        "stability": str(fact.get("stability") or "unknown"),
        "frequency": frequency_bucket(int(freq)),
    }


def _index_probes(probes: Iterable[Probe | dict]) -> dict[str, dict[str, Any]]:
    return {str(as_dict(p).get("probe_id", "")): as_dict(p) for p in probes}


def _index_sessions(sessions: Iterable[Session | dict]) -> dict[str, dict[str, Any]]:
    return {str(as_dict(s).get("session_id", "")): as_dict(s) for s in sessions}


def _facts_by_id(
    session_index: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """``(session_id, fact_id) -> fact row`` across all sessions.

    The composite key matters: ``fact_id`` is only unique *within* a session, so
    keying on it alone collapses every session's ``f01`` into one entry and makes
    the survival table silently report on a handful of facts instead of all of
    them.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for session_id, session in session_index.items():
        for fact in session.get("facts") or []:
            fact_id = str(fact.get("fact_id", ""))
            if fact_id:
                out[(str(session_id), fact_id)] = fact
    return out


def _compressed_texts(
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult] | None,
) -> dict[str, dict[str, str]]:
    """``{strategy: {session_id: compressed_text}}``."""
    out: dict[str, dict[str, str]] = {}
    if not compressions:
        return out
    if isinstance(compressions, dict):
        items = [r for rows in compressions.values() for r in (rows or [])]
    else:
        items = list(compressions)
    for raw in items:
        row = as_dict(raw)
        strategy = str(row.get("strategy") or "")
        session_id = str(row.get("session_id") or "")
        if strategy and session_id:
            out.setdefault(strategy, {})[session_id] = str(row.get("text") or "")
    return out


def _compressed_token_counts(
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult] | None,
) -> dict[tuple[str, str], int]:
    """``{(strategy, session_id): compressed_tokens}``, precomputed for the case records.

    Built once instead of being rescanned per lost case: a full run has hundreds of lost
    probes and a few hundred compressions, so the naive nested scan would be quadratic
    for no benefit.
    """
    out: dict[tuple[str, str], int] = {}
    if not compressions:
        return out
    items = (
        [r for rows in compressions.values() for r in (rows or [])]
        if isinstance(compressions, dict)
        else list(compressions)
    )
    for raw in items:
        row = as_dict(raw)
        strategy = str(row.get("strategy") or "")
        session_id = str(row.get("session_id") or "")
        if strategy and session_id:
            out[(strategy, session_id)] = int(row.get("compressed_tokens") or 0)
    return out


def _survives(value: str, text: str) -> bool:
    """Case/whitespace-insensitive substring check for the fact's canonical value."""
    needle = _norm(value)
    if not needle:
        return False
    return needle in _norm(text)


def _snippet(text: str, value: str, *, window: int = 90) -> str:
    """A short window around the fact's value in the compressed text (or the head).

    Used for the human-readable explanation. When the value is absent we show the start
    of the compressed text, which is what the reader needs to see: the memory's opening
    is often all a summariser kept.
    """
    if not text:
        return ""
    needle = _norm(value)
    if needle:
        idx = _norm(text).find(needle)
        if idx >= 0:
            # Map back approximately: normalised text is shorter or equal, so clamp.
            start = max(0, min(idx, len(text) - 1) - window // 2)
            return text[start : start + window].replace("\n", " ").strip()
    return text[:window].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Pairing full vs compressed answers
# ---------------------------------------------------------------------------
def _pair_conditions(
    rows: list[AnswerRow],
) -> dict[tuple[str, str], dict[str, AnswerRow]]:
    """Pair the two conditions per ``(probe_id, strategy)``.

    Returns ``{(probe_id, strategy): {"full": row, "compressed": row}}``. The baseline
    answer for a probe is shared across strategies (it is the same full-context call),
    which is exactly what makes "lost under compression" a clean within-probe contrast:
    the same question, the same model, only the memory differs.
    """
    pairs: dict[tuple[str, str], dict[str, AnswerRow]] = {}
    full_rows: dict[str, AnswerRow] = {}
    for row in rows:
        if str(row.get("strategy") or "") == FULL_STRATEGY:
            full_rows[str(row.get("probe_id"))] = row
    for row in rows:
        strategy = str(row.get("strategy") or FULL_STRATEGY)
        if strategy == FULL_STRATEGY:
            continue
        key = (str(row.get("probe_id")), strategy)
        slot = pairs.setdefault(key, {})
        slot["compressed"] = row
    for (probe_id, strategy), slot in pairs.items():
        if probe_id in full_rows:
            slot["full"] = full_rows[probe_id]
    return pairs


def _judged_ok(row: AnswerRow | None) -> bool:
    return bool(row) and not row.get("error") and row.get("correct") is not None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _new_bucket() -> dict[str, int]:
    return {"n": 0, "n_survived": 0}


def _finalise_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    n = bucket["n"]
    return {
        "n": n,
        "n_survived": bucket["n_survived"],
        "survival_rate": round(bucket["n_survived"] / n, 6) if n else None,
    }


def _empty_survival() -> dict[str, dict[str, dict[str, int]]]:
    """Pre-seeded accumulator so every known attribute value appears in the output.

    A value with ``n=0`` is information ("no deep_buried fact was probed"), and the
    report's charts need the key to exist. Seeding also makes the JSON shape stable
    across runs.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    for group in ATTRIBUTE_GROUPS:
        out[group] = {value: _new_bucket() for value in _ATTRIBUTE_VALUES[group]}
    return out


def _survival_from_fact_states(
    fact_states: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-fact survival booleans into the report's attribute table.

    ``fact_states`` maps ``fact_id -> {"attributes": {...}, "survived": bool}``. Only
    facts present in it are in the denominator (that is the "(a) were probed" clause of
    the module docstring).

    **Empty buckets and empty groups are omitted, not emitted as ``None``.** A bucket
    with ``n == 0`` means "no fact with this attribute value was probed", which is an
    absence of data rather than a 0% survival rate. Emitting ``survival_rate: None``
    would be truthful but would force every consumer (notably the report's charts) to
    null-check before plotting; omitting the key keeps the table a dense
    ``{value: {n, survival_rate}}`` map where every present rate is a real float. A
    group with no probed facts at all is dropped entirely, so the degenerate "no data"
    case yields ``{}`` — which the report renders as "no chart" rather than crashing on
    an empty subplot grid.
    """
    acc = _empty_survival()
    for state in fact_states.values():
        survived = bool(state["survived"])
        for group, value in state["attributes"].items():
            bucket = acc.setdefault(group, {}).setdefault(value, _new_bucket())
            bucket["n"] += 1
            if survived:
                bucket["n_survived"] += 1
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for group, values in acc.items():
        filled = {
            value: _finalise_bucket(b)
            for value, b in sorted(values.items())
            if b["n"] > 0
        }
        if filled:
            out[group] = filled
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_failures(
    answers: Iterable[Answer | AnswerRow],
    probes: Iterable[Probe | dict],
    sessions: Iterable[Session | dict],
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult] | None,
    *,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Join flipped probes to their facts and aggregate survival by fact attribute.

    A **lost probe** is one answered *correctly* under full context and *incorrectly*
    (or not at all) under a compressed condition. Probes that were already wrong with
    full context are excluded — they say nothing about compression. Probes whose
    compressed answer could not be judged are excluded from ``n_compared`` rather than
    being counted as lost, so gateway noise cannot inflate the loss rate.

    Parameters
    ----------
    answers
        Answer rows (``Answer`` objects or ``answers.jsonl`` dicts). Duplicates from
        retried cells are collapsed first, matching ``cost.compute_metrics``.
    probes, sessions, compressions
        Ground truth and the compressed memories, as produced by ``run.py``.
    out_dir
        When given, ``failures.json`` is written there.

    Returns
    -------
    dict
        ``summary`` (counts), ``survival_by_attribute`` (all strategies pooled),
        ``survival_by_attribute_by_strategy``, ``loss_by_kind``, ``cases`` (one per lost
        probe, with the verbatim-survival check per fact) and ``lost_probe_ids``.
    """
    rows = list(dedupe_answers(answers).values())
    probe_index = _index_probes(probes)
    session_index = _index_sessions(sessions)
    fact_index = _facts_by_id(session_index)
    compressed = _compressed_texts(compressions)
    compressed_counts = _compressed_token_counts(compressions)

    pairs = _pair_conditions(rows)

    cases: list[dict[str, Any]] = []
    lost_ids: list[str] = []
    n_compared = 0
    loss_by_kind: dict[str, dict[str, int]] = {}
    # (session_id, fact_id) -> pooled state across strategies (survived if it
    # survived in ANY arm).
    #
    # The composite key is load-bearing: ``fact_id`` is only unique *within* a
    # session ("f01" exists in every session), so keying on it alone silently
    # collapsed 30 sessions' facts into 9 and made every survival rate 1.0.
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    per_strategy_states: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    tracked_facts: set[tuple[str, str]] = set()
    lost_verbatim_facts: set[tuple[str, str]] = set()

    for (probe_id, strategy), slot in sorted(pairs.items()):
        full_row = slot.get("full")
        comp_row = slot.get("compressed")
        if not _judged_ok(full_row) or not _judged_ok(comp_row):
            continue
        n_compared += 1

        probe = probe_index.get(probe_id, {})
        kind = str(probe.get("kind") or "unknown")
        kind_slot = loss_by_kind.setdefault(kind, {"n_compared": 0, "n_lost": 0})
        kind_slot["n_compared"] += 1

        session_id = str(probe.get("session_id") or comp_row.get("session_id") or "")
        text = compressed.get(strategy, {}).get(session_id, "")
        fact_ids = [str(f) for f in (probe.get("fact_ids") or [])]

        # Per-fact verbatim check, for every probed fact (lost or not): this is what
        # makes the survival table's denominator "probed critical facts".
        fact_records: list[dict[str, Any]] = []
        for fact_id in fact_ids:
            fact = fact_index.get((session_id, fact_id))
            if fact is None:
                continue
            value = str(fact.get("value") or "")
            survived = _survives(value, text)
            attrs = _fact_attributes(fact)
            key = (session_id, fact_id)
            tracked_facts.add(key)
            if not survived:
                lost_verbatim_facts.add(key)

            state = pooled.setdefault(
                key, {"attributes": attrs, "survived": False, "fact": fact}
            )
            state["survived"] = bool(state["survived"]) or survived
            strat_state = per_strategy_states.setdefault(strategy, {}).setdefault(
                key, {"attributes": attrs, "survived": False, "fact": fact}
            )
            strat_state["survived"] = bool(strat_state["survived"]) or survived

            fact_records.append(
                {
                    "fact_id": fact_id,
                    "value": value,
                    "kind": str(fact.get("kind") or ""),
                    "frequency": int(
                        fact.get("frequency")
                        if fact.get("frequency") is not None
                        else len(fact.get("mentions") or [])
                    ),
                    "salience": attrs["salience"],
                    "embedding": attrs["embedding"],
                    "stability": attrs["stability"],
                    "survived_verbatim": survived,
                }
            )

        is_lost = full_row.get("correct") is True and comp_row.get("correct") is not True
        if not is_lost:
            continue

        kind_slot["n_lost"] += 1
        lost_ids.append(probe_id)
        missing = [f["fact_id"] for f in fact_records if not f["survived_verbatim"]]
        cases.append(
            {
                "probe_id": probe_id,
                "kind": kind,
                "strategy": strategy,
                "session_id": session_id,
                "question": str(probe.get("question") or ""),
                "reference_answer": str(probe.get("reference_answer") or ""),
                "answer_text": str(comp_row.get("answer_text") or ""),
                "full_context_answer": str(full_row.get("answer_text") or ""),
                "facts": fact_records,
                "n_facts_verbatim_missing": len(missing),
                "compressed_tokens": compressed_counts.get((strategy, session_id), 0),
                "compressed_snippet": _snippet(
                    text, fact_records[0]["value"] if fact_records else ""
                ),
                "explanation": _explain(probe, fact_records, comp_row),
            }
        )

    survival_all = _survival_from_fact_states(pooled)
    survival_by_strategy = {
        strategy: _survival_from_fact_states(states)
        for strategy, states in sorted(per_strategy_states.items())
    }

    doc: dict[str, Any] = {
        "summary": {
            "n_lost": len(cases),
            "n_compared": n_compared,
            "loss_rate": round(len(cases) / n_compared, 6) if n_compared else None,
            "n_facts_lost_verbatim": len(lost_verbatim_facts),
            "n_facts_tracked": len(tracked_facts),
            "n_lost_probes": len(set(lost_ids)),
            "lost_probe_ids": sorted(set(lost_ids)),
        },
        "survival_by_attribute": survival_all,
        "survival_by_attribute_by_strategy": survival_by_strategy,
        "loss_by_kind": {
            kind: {
                "n_compared": v["n_compared"],
                "n_lost": v["n_lost"],
                "loss_rate": round(v["n_lost"] / v["n_compared"], 6) if v["n_compared"] else None,
            }
            for kind, v in sorted(loss_by_kind.items())
        },
        "cases": cases,
        "definitions": {
            "lost_probe": (
                "correct under full context AND not correct under the compressed "
                "condition (for the same probe and strategy)"
            ),
            "verbatim_survival": (
                "the fact's canonical value appears as a case- and whitespace-"
                "insensitive substring of the compressed text"
            ),
            "survival_rate": (
                "of the distinct critical facts with a given attribute value that are "
                "referenced by at least one probe evaluated in both conditions, the "
                "fraction whose value still appears verbatim in the compressed memory"
            ),
            "denominator_note": (
                "the denominator is probed facts, not all facts; a fact no probe asks "
                "about cannot be scored"
            ),
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc


def _explain(probe: dict[str, Any], facts: list[dict[str, Any]], comp_row: AnswerRow) -> str:
    """One-sentence, human-readable account of a single lost probe.

    The explanation names the mechanism the data supports rather than guessing: whether
    the value is missing from the memory (dropped), present but unused (retrieval
    failure), or the answer contradicts it (superseded-trap failure).
    """
    if not facts:
        return (
            "The probe flipped from correct to incorrect under compression, but its "
            "facts are not in the dataset index (join miss)."
        )
    missing = [f for f in facts if not f["survived_verbatim"]]
    present = [f for f in facts if f["survived_verbatim"]]
    answer = _norm(str(comp_row.get("answer_text") or ""))
    answer_empty = not answer or answer.startswith("i don't know") or "don't know" in answer

    if missing and not present:
        f = missing[0]
        return (
            f"The compressed memory dropped `{f['value']}` entirely ({f['embedding']}, "
            f"{f['salience']}, mentioned {f['frequency']}x"
            f"{', superseded' if f['stability'] == 'superseded' else ''}), so the answerer "
            f"had no way to retrieve it."
        )
    if missing and present:
        return (
            f"Partially kept: {len(present)} of {len(facts)} fact values survived verbatim "
            f"but `{missing[0]['value']}` did not, so a {str(probe.get('kind') or 'multi-hop')} "
            f"question needing all of them failed."
        )
    if answer_empty:
        return (
            "Every fact value is still present verbatim in the compressed memory, but the "
            "answerer refused to answer — a retrieval/formatting failure rather than a "
            "content loss."
        )
    return (
        "Every fact value is still present verbatim in the compressed memory, but the "
        "answerer produced a wrong or outdated value: the memory kept the fact and the "
        "model failed to use it (or a superseded mention won)."
    )


__all__ = ["analyze_failures", "frequency_bucket"]
