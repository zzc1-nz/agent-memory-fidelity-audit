"""Metrics layer for the memory-compression fidelity audit.

Sub-modules
-----------
judge      : LLM-as-judge correctness grading (independent model from the answerer).
evaluate   : the probe x condition matrix (full context vs compressed), resumable.
cost       : the headline metric — tokens spent per *correct* answer — plus the
             compression-ratio/accuracy/breakeven comparison. Writes ``metrics.json``.
failures   : which facts a compressor dropped, joined back to their ground-truth
             attributes. Writes ``failures.json``.

Why this package exists
-----------------------
The compression literature reports *how small* a memory got. This package reports
*how expensive a correct answer got* once compression had its way with the history.
Everything here is a pure function of the data objects passed in (plus one LLM call
per judge verdict), so it can be unit-tested offline and re-run on cached rows.

Shared answer-loading helpers
-----------------------------
``answers.jsonl`` is append-only and resumable, so the same
``(probe_id, strategy, condition)`` cell can legitimately appear more than once after
a retry. ``dedupe_answers`` is the single definition of "the row we keep", used by
both ``cost`` and ``failures`` so the two reports can never disagree about which
answers they are describing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from memfaith.schema import Answer

from .judge import judge_answer, parse_verdict

__all__ = [
    "AnswerRow",
    "analyze_failures",
    "answer_key",
    "as_dict",
    "compute_metrics",
    "dedupe_answers",
    "judge_answer",
    "load_answers",
    "parse_verdict",
    "run_matrix",
    "write_answers_jsonl",
]

# An answer at rest is a plain dict (``Answer.to_dict()``). In memory it may still be
# the dataclass, so every public entry point normalises first.
AnswerRow = dict[str, Any]

FULL_STRATEGY = "full_context"
FULL_CONDITION = "full"
COMPRESSED_CONDITION = "compressed"


def as_dict(obj: Any) -> dict[str, Any]:
    """Return a plain ``dict`` for a dataclass row or pass a mapping through.

    The experiment writes JSONL and re-reads it, so every consumer in this package
    has to accept both shapes. Keeping the normalisation in one place avoids the
    classic bug where one module indexes an object and the other a dict.
    """
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # Dataclasses from schema.py all expose ``__dict__``; ``asdict`` is not used
    # because nested dataclasses must already have been serialised by the caller.
    return dict(vars(obj))


def answer_key(row: AnswerRow | Answer) -> tuple[str, str, str]:
    """Identity of an evaluation cell: ``(probe_id, strategy, condition)``.

    ``strategy`` is ``"full_context"`` for the baseline condition, so the full-context
    arm and each compressed arm are distinct keys even for the same probe.
    """
    d = as_dict(row)
    return (
        str(d.get("probe_id", "")),
        str(d.get("strategy") or FULL_STRATEGY),
        str(d.get("condition") or FULL_CONDITION),
    )


def load_answers(path: str) -> list[AnswerRow]:
    """Read ``answers.jsonl``, tolerating a truncated final line.

    A hard kill mid-``write`` can leave a partial last line. Dropping it is correct:
    that row was never completed, so it will be recomputed on resume. Corruption
    anywhere *else* is a real error and is raised, because silently skipping a middle
    row would hide data loss.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    rows: list[AnswerRow] = []
    last = len(lines) - 1
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == last:
                break  # torn write from a crash: the cell will be recomputed
            raise
    return rows


def write_answers_jsonl(path: str, rows: Iterable[AnswerRow | Answer]) -> None:
    """Overwrite ``path`` with ``rows`` (used by tests and repair tooling)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(as_dict(row), ensure_ascii=False) + "\n")


def dedupe_answers(rows: Iterable[AnswerRow | Answer]) -> dict[tuple[str, str, str], AnswerRow]:
    """Collapse duplicate cells, preferring a successful row over an errored one.

    ``answers.jsonl`` is append-only: if a cell failed on a flaky gateway and was
    retried on the next run, both rows are on disk. The retried success must win, or
    the metrics would silently count a transient network error as an incorrect
    answer. When both rows are equally good (or equally bad) the later one wins,
    which makes the result independent of how many times a cell was retried.
    """
    best: dict[tuple[str, str, str], AnswerRow] = {}
    for raw in rows:
        row = as_dict(raw)
        key = answer_key(row)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        # An errored row is strictly worse than a clean one; otherwise last write wins.
        if current.get("error") and not row.get("error"):
            best[key] = row
        elif bool(current.get("error")) == bool(row.get("error")):
            best[key] = row
    return best


def judged(row: AnswerRow) -> bool:
    """True when the row has a usable verdict (the judge actually produced one)."""
    return not row.get("error") and row.get("correct") is not None


# Imported last so the sub-modules can import the helpers above without a cycle.
from .evaluate import run_matrix  # noqa: E402
from .cost import compute_metrics  # noqa: E402
from .failures import analyze_failures  # noqa: E402
