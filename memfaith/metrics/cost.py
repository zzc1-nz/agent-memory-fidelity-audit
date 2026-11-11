"""The headline metric: tokens spent per *correct* answer, and the breakeven condition.

The thesis
----------
Compression papers report **ratio**: how much smaller the memory got. That number is
not the objective. The objective is a working agent, and the thing an agent's operator
actually pays for is a *correct answer*. So the audit's primary metric is:

    cost_per_correct_answer = total_tokens_spent / number_of_correct_answers

A compressor that cuts the context by 75% but drops 40% of the answers raises that
number. It looks like a win in every table in the literature and is a loss in practice.
This module computes the number both ways:

``total_tokens_one_shot``
    Just the compressed context: what a *single* query costs after the memory has
    already been built. This is the number a compression paper implicitly reports.

``total_tokens_amortized``
    Compressed context **plus** the tokens spent producing the compression
    (``CompressionResult.cost_tokens``). Building a summary is not free; if a session
    is consulted even once, that spend belongs in the denominator. This is the honest
    number and the one the report leads with.

Both are divided by the *correct* answer count, so a cheaper-but-worse memory cannot
win by being cheap. When a arm gets nothing right, the ratio is undefined: the guard
returns ``None`` and sets ``zero_correct=True`` rather than raising or returning
``inf``, because both of those would silently corrupt a downstream average.

The breakeven derivation
------------------------
With ``R`` = token ratio (compressed/full), ``A_c``/``A_f`` = accuracy under each
condition, and ``P`` probes per condition:

    cost_c < cost_f
    <=>  (R * T_f) / (A_c * P) < T_f / (A_f * P)
    <=>  R / A_c < 1 / A_f
    <=>  A_c / A_f > R                     # retention must exceed the token ratio

i.e. **compression pays off iff the fraction of accuracy it preserves is larger than
the fraction of tokens it saves**. ``max_viable_ratio`` reports the retention, which is
the largest compression ratio that would still have broken even: if you achieved
``ratio=0.25`` but only retained ``0.60`` of the accuracy, any ratio above ``0.60`` was
a net loss and the measured ``0.25`` was far past the cliff edge. Including the
compression spend replaces ``R`` with the effective ratio ``(CT_c + CC_c) / CT_f``;
both variants are emitted (``ratio`` / ``ratio_amortized``).

Attribution rules (documented because they move the numbers)
------------------------------------------------------------
* ``context_tokens_total`` sums ``Answer.context_tokens`` — ``count_tokens`` of the exact
  context string that was sent — over every cell whose answerer call was actually made.
  A cell that failed before reaching the model contributes nothing; a cell that answered
  but could not be judged still consumed context and *does* count.
* ``n_probes`` is the number of **judged** cells (the accuracy denominator). Unjudged
  cells (gateway failures) are reported separately as ``n_unjudged`` and never counted
  as wrong — an infrastructure failure is not evidence about compression.
* ``n_partial`` counts judge verdicts of ``partial``. They are **not** correct for the
  headline metric (a half-recovered fact is not usable), but they are shown so the report
  can distinguish "lost" from "nearly kept".
* Per-kind compression cost is allocated by that kind's share of the arm's context
  tokens, because compression spend is per session while accuracy is per probe.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from memfaith.schema import Answer, CompressionResult, Probe, Session

from . import (
    FULL_STRATEGY,
    AnswerRow,
    as_dict,
    dedupe_answers,
)
from .judge import verdict_label

#: A judge verdict of ``partial`` is not a correct answer, but it is worth showing.
PARTIAL = "partial"
UNPARSED = "unparsed"

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _r(value: float | None, ndigits: int = 6) -> float | None:
    """Round for stable, diff-friendly JSON; pass ``None`` through."""
    if value is None:
        return None
    return round(float(value), ndigits)


def _ratio(num: float, den: float) -> float | None:
    """``num / den`` with a zero-denominator guard returning ``None``."""
    if den is None or abs(den) < _EPS:
        return None
    return num / den


def _index_probes(probes: Iterable[Probe | dict]) -> dict[str, dict[str, Any]]:
    return {str(as_dict(p).get("probe_id", "")): as_dict(p) for p in probes}


def _index_sessions(sessions: Iterable[Session | dict]) -> dict[str, dict[str, Any]]:
    return {str(as_dict(s).get("session_id", "")): as_dict(s) for s in sessions}


def _group_compressions(
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Normalise compressions to ``{strategy: [row, ...]}``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not compressions:
        return grouped
    if isinstance(compressions, dict):
        for strategy, rows in compressions.items():
            for row in rows or []:
                d = as_dict(row)
                d.setdefault("strategy", strategy)
                grouped.setdefault(str(strategy), []).append(d)
    else:
        for row in compressions:
            d = as_dict(row)
            grouped.setdefault(str(d.get("strategy", "")), []).append(d)
    return grouped


def _counts_toward_tokens(row: AnswerRow) -> bool:
    """Did this cell actually send a context to the answerer (i.e. spend tokens)?

    Permanent structural errors (no compression to send) never reached the model. Every
    other cell — including one whose *judge* call failed — did send a prompt and burned
    context tokens, so it belongs in the cost numerator.
    """
    error = str(row.get("error") or "")
    if not error:
        return True
    return error.startswith("judge_error")


def _is_judged(row: AnswerRow) -> bool:
    """A cell with a usable verdict (the accuracy denominator)."""
    return not row.get("error") and row.get("correct") is not None


def _label(row: AnswerRow) -> str:
    """``correct`` / ``incorrect`` / ``partial`` / ``unparsed`` for a row."""
    return verdict_label(str(row.get("judge_raw") or ""), row.get("correct"))


def _arm_rows(rows: Iterable[AnswerRow]) -> dict[str, list[AnswerRow]]:
    """Group answer rows by arm name (``full_context`` for the baseline)."""
    arms: dict[str, list[AnswerRow]] = {}
    for row in rows:
        arm = str(row.get("strategy") or FULL_STRATEGY)
        arms.setdefault(arm, []).append(row)
    return arms


# ---------------------------------------------------------------------------
# Per-arm block
# ---------------------------------------------------------------------------
def _accuracy(rows: list[AnswerRow]) -> tuple[int, int]:
    judged = [r for r in rows if _is_judged(r)]
    correct = sum(1 for r in judged if r.get("correct") is True)
    return correct, len(judged)


def _accuracy_by_kind(rows: list[AnswerRow], probes: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """Accuracy per probe kind, over judged cells only."""
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        if not _is_judged(row):
            continue
        probe = probes.get(str(row.get("probe_id")), {})
        kind = str(probe.get("kind") or "unknown")
        buckets.setdefault(kind, []).append(bool(row.get("correct")))
    return {kind: _r(_ratio(sum(v), len(v))) for kind, v in sorted(buckets.items())}


def _compression_ratio(strategy: str, comp_rows: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]) -> float | None:
    """Mean per-session ``compressed / original``. Baseline is exactly 1.0.

    Averaged per session rather than over the token total so one long session cannot
    define the arm's ratio. Falls back to the tokens recorded on the row when ``ratio``
    is missing (e.g. a hand-built fixture).
    """
    if strategy == FULL_STRATEGY:
        return 1.0
    if not comp_rows:
        return None
    ratios: list[float] = []
    for row in comp_rows:
        ratio = row.get("ratio")
        if ratio is None:
            comp_tokens = float(row.get("compressed_tokens") or 0)
            orig = float(row.get("original_tokens") or 0)
            ratio = _ratio(comp_tokens, orig)
        if ratio is not None:
            ratios.append(float(ratio))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _arm_block(
    *,
    strategy: str,
    rows: list[AnswerRow],
    probes: dict[str, dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    comp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute one arm's metrics block (the ``arms[name]`` object in metrics.json)."""
    judged = [r for r in rows if _is_judged(r)]
    n_correct = sum(1 for r in judged if r.get("correct") is True)
    n_partial = sum(1 for r in judged if _label(r) == PARTIAL)
    n_unparsed = sum(1 for r in rows if _label(r) == UNPARSED)
    n_probes = len(judged)

    context_tokens_total = sum(int(r.get("context_tokens") or 0) for r in rows if _counts_toward_tokens(r))
    compression_cost_tokens = 0 if strategy == FULL_STRATEGY else sum(
        int(r.get("cost_tokens") or 0) for r in comp_rows
    )
    total_amortized = context_tokens_total + compression_cost_tokens
    total_one_shot = context_tokens_total

    zero_correct = n_correct == 0
    cpa_amortized = None if zero_correct else total_amortized / n_correct
    cpa_one_shot = None if zero_correct else total_one_shot / n_correct

    return {
        "strategy": strategy,
        "compression_ratio": _r(_compression_ratio(strategy, comp_rows, sessions)),
        "context_tokens_total": context_tokens_total,
        "compression_cost_tokens": compression_cost_tokens,
        "total_tokens_amortized": total_amortized,
        "total_tokens_one_shot": total_one_shot,
        "n_probes": n_probes,
        "n_correct": n_correct,
        "accuracy": _r(_ratio(n_correct, n_probes)),
        "n_partial": n_partial,
        "n_unparsed": n_unparsed,
        "n_cells": len(rows),
        "cost_per_correct_answer_amortized": _r(cpa_amortized),
        "cost_per_correct_answer_one_shot": _r(cpa_one_shot),
        "zero_correct": zero_correct,
        "accuracy_by_kind": _accuracy_by_kind(rows, probes),
        # Filled in once the baseline is known (see _add_baseline_deltas).
        "cost_delta_vs_baseline_abs": None,
        "cost_delta_vs_baseline_pct": None,
        "cost_ratio_vs_baseline": None,
        "accuracy_delta_vs_baseline": None,
    }


def _add_baseline_deltas(arms: dict[str, dict[str, Any]]) -> None:
    """Attach baseline-relative deltas. Positive = compression costs *more*.

    ``cost_delta_vs_baseline_pct`` is a **fraction** (0.42 = +42%), not percentage
    points, because the report's formatter multiplies by 100. The absolute delta is
    emitted alongside so nothing is lost in the naming.
    """
    baseline = arms.get(FULL_STRATEGY)
    if not baseline:
        return
    base_amort = baseline.get("cost_per_correct_answer_amortized")
    base_oneshot = baseline.get("cost_per_correct_answer_one_shot")
    base_acc = baseline.get("accuracy")
    for name, block in arms.items():
        if name == FULL_STRATEGY:
            block["cost_delta_vs_baseline_abs"] = 0.0
            block["cost_delta_vs_baseline_pct"] = 0.0
            block["cost_ratio_vs_baseline"] = 1.0
            block["accuracy_delta_vs_baseline"] = 0.0
            continue
        cpa = block.get("cost_per_correct_answer_amortized")
        if cpa is not None and base_amort:
            block["cost_delta_vs_baseline_abs"] = _r(cpa - base_amort)
            block["cost_delta_vs_baseline_pct"] = _r((cpa - base_amort) / base_amort)
            block["cost_ratio_vs_baseline"] = _r(cpa / base_amort)
        else:
            block["cost_delta_vs_baseline_abs"] = None
            block["cost_delta_vs_baseline_pct"] = None
            block["cost_ratio_vs_baseline"] = None
        acc = block.get("accuracy")
        if acc is not None and base_acc is not None:
            block["accuracy_delta_vs_baseline"] = _r(acc - base_acc)
        else:
            block["accuracy_delta_vs_baseline"] = None
        # One-shot delta is also useful but the report only shows amortized.
        oneshot = block.get("cost_per_correct_answer_one_shot")
        if oneshot is not None and base_oneshot:
            block["cost_delta_vs_baseline_one_shot_pct"] = _r((oneshot - base_oneshot) / base_oneshot)
        else:
            block["cost_delta_vs_baseline_one_shot_pct"] = None


# ---------------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------------
def _breakeven_for(block: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate the breakeven condition for one compression arm.

    Returns ``ratio`` (measured), ``accuracy_retention`` (A_c / A_f) and the verdict:

    * ``compression_pays_off``  — retention > ratio: the accuracy kept is worth more
      than the tokens saved, so cost per correct answer fell.
    * ``compression_costs_more`` — retention <= ratio: the memory got cheaper and the
      answers got worse by more.
    * ``inconclusive`` — one of the two quantities is undefined (e.g. the baseline got
      nothing right, so retention has no meaning), or the arm produced no judged cells.

    Retention is computed from the **raw correct/judged counts**, not from the rounded
    ``accuracy`` fields. At the threshold the verdict hinges on a strict inequality, and
    rounding both accuracies to 6 dp first would turn an exact tie into a spurious win
    or loss (e.g. 3/9 vs 6/9 gives 0.499999 instead of exactly 0.5).
    """
    ratio = block.get("compression_ratio")
    n_correct = block.get("n_correct")
    n_probes = block.get("n_probes")
    base_correct = (baseline or {}).get("n_correct")
    base_probes = (baseline or {}).get("n_probes")

    acc = _ratio(n_correct, n_probes) if n_correct is not None and n_probes else None
    base_acc = _ratio(base_correct, base_probes) if base_correct is not None and base_probes else None
    retention = _ratio(acc, base_acc) if acc is not None and base_acc is not None else None

    if ratio is None or retention is None:
        verdict = "inconclusive"
    elif retention > float(ratio) + _EPS:
        verdict = "compression_pays_off"
    else:
        verdict = "compression_costs_more"

    # Effective (amortized) ratio: tokens actually spent to produce one compressed
    # context, including the one-time compression spend, relative to the full context.
    base_tokens = (baseline or {}).get("context_tokens_total") or 0
    amort_ratio = _ratio(
        (block.get("context_tokens_total") or 0) + (block.get("compression_cost_tokens") or 0),
        base_tokens,
    )

    # Largest ratio that still breaks even. The algebra gives `retention`, but a token
    # ratio is bounded by 1.0: if compression *improved* accuracy (retention > 1) then
    # any achievable ratio up to and including 1.0 would have paid off, and reporting
    # "max viable ratio = 300%" would be nonsense in the report. The uncapped retention
    # is still available as `accuracy_retention`.
    max_viable = None if retention is None else min(1.0, retention)
    max_viable_amort = None
    if retention is not None and base_tokens:
        max_viable_amort = min(1.0, retention - (block.get("compression_cost_tokens") or 0) / base_tokens)

    return {
        "strategy": block.get("strategy"),
        "ratio": _r(ratio),
        "accuracy_retention": _r(retention),
        "accuracy_compressed": acc,
        "accuracy_full": base_acc,
        "verdict": verdict,
        "max_viable_ratio": _r(max_viable),
        "ratio_amortized": _r(amort_ratio),
        "max_viable_ratio_amortized": _r(max_viable_amort),
        "cost_ratio_vs_baseline": block.get("cost_ratio_vs_baseline"),
        "token_saving_ratio": _r(1.0 - ratio) if ratio is not None else None,
    }


def _pick_headline_arm(arms: dict[str, dict[str, Any]], cfg: Any) -> str | None:
    """Choose the arm the top-level ``breakeven`` block describes.

    The most *charitable* compression arm — the one with the lowest amortized cost per
    correct answer (ties broken by configured order) — is used for the headline. That
    makes the claim conservative: if even the best compression arm costs more per correct
    answer than the baseline, the finding holds for the whole design space, not just for
    a weak arm. The chosen arm is named in the output so the report can attribute it.
    """
    configured = [s for s in getattr(cfg, "strategies", ()) if s in arms]
    candidates = [s for s in configured if s != FULL_STRATEGY]
    if not candidates:
        candidates = [s for s in arms if s != FULL_STRATEGY]
    if not candidates:
        return None

    def rank(name: str) -> tuple[int, float]:
        cpa = arms[name].get("cost_per_correct_answer_amortized")
        # Arms with no defined cost sort last, in configured order.
        return (1, 0.0) if cpa is None else (0, float(cpa))

    order = {name: i for i, name in enumerate(candidates)}
    return min(candidates, key=lambda n: (rank(n), order[n]))


# ---------------------------------------------------------------------------
# Per-probe-kind cost breakdown
# ---------------------------------------------------------------------------
def _per_kind_cost(
    arms: dict[str, dict[str, Any]],
    arm_rows: dict[str, list[AnswerRow]],
    probes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Cost per correct answer *within each probe kind*.

    Compression spend is per session, so it is allocated to a kind by that kind's share
    of the arm's context tokens. The alternative (splitting it evenly across kinds)
    would misprice kinds that read much less context. ``n_correct == 0`` yields ``None``
    for that kind's cost, never an exception.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, block in arms.items():
        rows = arm_rows.get(arm, [])
        total_ctx = sum(int(r.get("context_tokens") or 0) for r in rows if _counts_toward_tokens(r))
        comp_cost = int(block.get("compression_cost_tokens") or 0)
        per_kind: dict[str, dict[str, Any]] = {}
        kinds: dict[str, list[AnswerRow]] = {}
        for row in rows:
            kind = str(probes.get(str(row.get("probe_id")), {}).get("kind") or "unknown")
            kinds.setdefault(kind, []).append(row)
        for kind, kind_rows in sorted(kinds.items()):
            judged = [r for r in kind_rows if _is_judged(r)]
            n_correct = sum(1 for r in judged if r.get("correct") is True)
            ctx = sum(int(r.get("context_tokens") or 0) for r in kind_rows if _counts_toward_tokens(r))
            share = _ratio(ctx, total_ctx) or 0.0
            alloc_cost = int(round(comp_cost * share))
            total = ctx + alloc_cost
            per_kind[kind] = {
                "n_probes": len(judged),
                "n_correct": n_correct,
                "accuracy": _r(_ratio(n_correct, len(judged))),
                "context_tokens_total": ctx,
                "compression_cost_tokens_allocated": alloc_cost,
                "total_tokens_amortized": total,
                "cost_per_correct_answer_amortized": _r(None if n_correct == 0 else total / n_correct),
                "zero_correct": n_correct == 0,
            }
        out[arm] = per_kind
    return out


# ---------------------------------------------------------------------------
# Verdict sentence
# ---------------------------------------------------------------------------
def _headline(arms: dict[str, dict[str, Any]], breakeven: dict[str, Any]) -> str:
    """One punchy sentence with the real measured numbers (report bottom line)."""
    base = arms.get(FULL_STRATEGY) or {}
    arm_name = breakeven.get("strategy")
    block = arms.get(str(arm_name)) or {}
    if not block or block.get("accuracy") is None or base.get("accuracy") is None:
        return ("No compression arm produced a measurable accuracy result, so the "
                "cost-per-correct-answer comparison is inconclusive.")

    ratio = block.get("compression_ratio")
    acc_c = block.get("accuracy")
    acc_f = base.get("accuracy")
    cpa_c = block.get("cost_per_correct_answer_amortized")
    cpa_f = base.get("cost_per_correct_answer_amortized")
    n_c, n_f = block.get("n_correct", 0), base.get("n_correct", 0)
    n_p = block.get("n_probes", 0)

    def pct(v: Any) -> str:
        return "n/a" if v is None else f"{v * 100:.0f}%"

    def toks(v: Any) -> str:
        return "n/a" if v is None else f"{v:,.0f}"

    if breakeven.get("verdict") == "compression_pays_off":
        retention = breakeven.get("accuracy_retention")
        # Retention above 100% means compression *helped* (usually because the summary is
        # easier to read than a long transcript). Saying "keeps 300% of the accuracy" is
        # accurate but reads as a bug, so phrase it as an improvement instead.
        if retention is not None and retention > 1.0:
            kept = (f"and answers *better* than full context "
                    f"({pct(acc_c)} vs {pct(acc_f)} accuracy), so each correct "
                    f"answer costs **{toks(cpa_c)}** tokens versus **{toks(cpa_f)}**")
        else:
            kept = (f"and keeps {pct(retention)} of the accuracy, so each correct "
                    f"answer costs **{toks(cpa_c)}** tokens versus **{toks(cpa_f)}**")
        lead = (f"`{arm_name}` compresses to {pct(ratio)} of the original tokens "
                f"{kept} for full context — compression pays off here.")
        if cpa_c is not None and cpa_f:
            lead += f" That is {abs((cpa_c - cpa_f) / cpa_f) * 100:.0f}% cheaper per correct answer."
        return lead

    if breakeven.get("verdict") == "compression_costs_more":
        ratio_txt = pct(ratio)
        retention_txt = pct(breakeven.get("accuracy_retention"))
        sentence = (
            f"`{arm_name}` shrinks the memory to {ratio_txt} of its tokens but keeps only "
            f"{retention_txt} of the answers ({n_c}/{n_p} correct vs {n_f} with full context), "
            f"so each correct answer costs **{toks(cpa_c)}** tokens instead of **{toks(cpa_f)}** "
            f"— the ratio improved and the economics got worse"
        )
        if cpa_c is not None and cpa_f:
            sentence += (f" ({cpa_c / cpa_f:.2f}x). The breakeven point was a ratio of "
                         f"{pct(breakeven.get('max_viable_ratio'))}; we were at {ratio_txt}.")
        else:
            sentence += "."
        return sentence

    return (f"`{arm_name}` reached {pct(acc_c)} accuracy at a ratio of {pct(ratio)}, but the "
            f"baseline accuracy was {pct(acc_f)}, so the breakeven comparison is inconclusive.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_metrics(
    answers: Iterable[Answer | AnswerRow],
    probes: Iterable[Probe | dict],
    sessions: Iterable[Session | dict],
    compressions: dict[str, list[CompressionResult]] | Iterable[CompressionResult] | None,
    cfg: Any,
    *,
    usage: dict | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Build the full metrics document (and optionally write ``metrics.json``).

    Parameters
    ----------
    answers
        Answer rows (dicts from ``answers.jsonl`` or ``Answer`` objects). Duplicates from
        retried cells are collapsed by :func:`memfaith.metrics.dedupe_answers`.
    probes, sessions
        Ground truth; only ``probe_id -> kind`` and ``session_id`` are needed here.
    compressions
        ``{strategy: [CompressionResult]}``. Provides the ratio and the tokens spent
        producing each compression (``cost_tokens``).
    cfg
        ``RunConfig``; ``strategies`` drives arm order and the headline arm choice.
    usage
        Optional ``LLMClient.usage()`` dict, embedded as ``token_usage`` for the report's
        "experiment token spend" line.
    out_dir
        When given, ``metrics.json`` is written there. ``run.py`` also writes the file
        itself; both produce the same document, so either path is safe.

    Returns
    -------
    dict
        The document described in the team contract: ``arms``, ``accuracy_by_kind``,
        ``breakeven``, ``verdict``, ``token_usage``, ``by_probe_kind_cost`` plus the
        ``when_is_compression_worth_it`` table.
    """
    rows = list(dedupe_answers(answers).values())
    probe_index = _index_probes(probes)
    session_index = _index_sessions(sessions)
    grouped = _group_compressions(compressions)
    by_arm = _arm_rows(rows)

    # Arm order: baseline first, then the configured strategies, then anything else the
    # data contains. Strategies present only in ``compressions`` (no answers yet) are
    # included too, so a partially completed run still reports the ratio it achieved
    # instead of silently dropping the arm.
    arm_order = [FULL_STRATEGY]
    for name in list(getattr(cfg, "strategies", ())) + list(grouped) + list(by_arm):
        if name not in arm_order:
            arm_order.append(name)

    arms: dict[str, dict[str, Any]] = {}
    for strategy in arm_order:
        arms[strategy] = _arm_block(
            strategy=strategy,
            rows=by_arm.get(strategy, []),
            probes=probe_index,
            sessions=session_index,
            comp_rows=grouped.get(strategy, []),
        )
    _add_baseline_deltas(arms)

    baseline = arms.get(FULL_STRATEGY)
    breakeven_by_strategy = {
        name: _breakeven_for(block, baseline)
        for name, block in arms.items()
        if name != FULL_STRATEGY
    }
    headline_arm = _pick_headline_arm(arms, cfg)
    breakeven = (
        dict(breakeven_by_strategy[headline_arm])
        if headline_arm in breakeven_by_strategy
        else {
            "strategy": None, "ratio": None, "accuracy_retention": None,
            "verdict": "inconclusive", "max_viable_ratio": None,
        }
    )

    worth_it = [
        {
            "strategy": name,
            "compression_ratio": be.get("ratio"),
            "accuracy": arms[name].get("accuracy"),
            "baseline_accuracy": (baseline or {}).get("accuracy"),
            "accuracy_retention": be.get("accuracy_retention"),
            "cost_per_correct_answer_amortized": arms[name].get("cost_per_correct_answer_amortized"),
            "baseline_cost_per_correct_answer": (baseline or {}).get("cost_per_correct_answer_amortized"),
            "cost_ratio_vs_baseline": be.get("cost_ratio_vs_baseline"),
            "breakeven_holds": be.get("verdict") == "compression_pays_off",
        }
        for name, be in breakeven_by_strategy.items()
        # A configured arm that produced no judged cells has nothing to say about the
        # trade-off; it stays in ``arms`` (with nulls) but not in this comparison table.
        if arms[name].get("accuracy") is not None
    ]

    doc: dict[str, Any] = {
        "arms": arms,
        "accuracy_by_kind": {name: block["accuracy_by_kind"] for name, block in arms.items()},
        "breakeven": breakeven,
        "breakeven_by_strategy": breakeven_by_strategy,
        "verdict": {"headline": _headline(arms, breakeven)},
        "token_usage": usage or {},
        "by_probe_kind_cost": _per_kind_cost(arms, by_arm, probe_index),
        "when_is_compression_worth_it": worth_it,
        "methodology": {
            "cost_per_correct_answer": (
                "total_tokens_spent / n_correct; amortized includes the tokens spent "
                "producing each compression, one_shot counts the compressed context only"
            ),
            "breakeven_condition": (
                "compression pays off iff accuracy_retention (A_comp/A_full) > "
                "compression_ratio (compressed/original tokens)"
            ),
            "n_probes_denominator": (
                "judged cells only; gateway/unparseable verdicts are excluded from "
                "accuracy, never counted as incorrect"
            ),
            "partial_policy": "verdict 'partial' is reported but is NOT correct",
            "baseline_arm": FULL_STRATEGY,
            "compression_ratio_baseline": 1.0,
            "delta_sign_convention": (
                "positive cost_delta_vs_baseline_* means compression is MORE expensive "
                "per correct answer than full context"
            ),
            "headline_arm": headline_arm,
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc


__all__ = ["compute_metrics"]
