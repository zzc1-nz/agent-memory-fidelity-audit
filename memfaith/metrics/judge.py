"""LLM-as-judge correctness grading — the measurement instrument of the audit.

Why an LLM judge, and why a *different* model
---------------------------------------------
The audit's headline number is "tokens spent per **correct** answer". Correctness has
to be measured per probe, and the probes' reference answers are short literal facts
(a path, a package name, a date), so a human rubric is easy to write but too slow to
apply thousands of times. An LLM judge applies that rubric at scale.

Using the answerer as its own judge would be circular: a model that hallucinates a
plausible path will also find that path plausible on review. ``config.MODELS``
therefore pins three distinct model identities and this module always asks for the
``judge`` role — never a literal model name. Swapping the gateway or the models is a
config change, not a code change.

The failure mode this module is built to survive
------------------------------------------------
Free-tier gateways return prose, markdown fences, trailing commentary, or an empty
string under load. A judge that raises on a malformed reply would kill a 300-call
run; a judge that silently defaults to "correct" would inflate accuracy and destroy
the finding. So parsing is layered and explicit:

  1. strip markdown code fences,
  2. locate the first balanced ``{...}`` object and ``json.loads`` it,
  3. fall back to keyword regexes over the raw text (negation-aware),
  4. otherwise return ``None`` — "no verdict" — and keep the raw text for re-audit.

``None`` is propagated all the way into ``Answer.correct`` and excluded from the
accuracy denominator, so an unparseable verdict can never be mistaken for a pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from memfaith.schema import Probe

# ---------------------------------------------------------------------------
# The rubric. Kept as a module constant so the report can quote it verbatim and so
# tests can assert on its content.
# ---------------------------------------------------------------------------
JUDGE_RULES = """\
You are a strict grader for a memory-fidelity benchmark.

An agent was given a memory of a past session (either the full transcript or a
compressed summary of it). It was then asked a question about that session and
produced the CANDIDATE ANSWER below. Your job is to decide whether the candidate
answer still carries the ground-truth fact.

QUESTION
{question}

REFERENCE FACT (ground truth — the canonical, current value)
{reference_answer}

CANDIDATE ANSWER
{answer}

GRADING RULES
1. The candidate is CORRECT only if it states the reference fact. Paraphrase and
   reordering are fine as long as the required value(s) are unambiguously present
   and right.
2. Extra information is fine. A longer answer that also contains the reference fact
   is correct; do not penalise verbosity or extra true statements.
3. A different, wrong, or OUTDATED value is INCORRECT — even if the answer is
   fluent, confident and plausible. If the question asks for the current value and
   the candidate gives a superseded one, that is incorrect.
4. A refusal, an "I don't know", or a hedge that does not commit to the fact
   ("possibly", "it might have been", "one of the paths was...") is INCORRECT. Do
   not give credit for uncertainty that happens to point near the right answer.
5. Use PARTIAL only when the candidate contains a genuine part of the reference
   fact but not all of it (e.g. right directory but wrong filename; two of three
   required values). PARTIAL is NOT correct.

Reply with ONE JSON object and nothing else:
{{"verdict": "correct" | "incorrect" | "partial", "confidence": <number between 0.0 and 1.0>}}

"confidence" is your confidence in the verdict you just gave (1.0 = certain)."""

#: The exact instruction block sent to the judge (before formatting). Exported so the
#: report and the team can quote the instrument rather than paraphrase it.
JUDGE_PROMPT_TEMPLATE = JUDGE_RULES

#: ``max_tokens`` for the judge call. Reasoning models spend hidden tokens before
#: emitting content and return an empty string if the budget is too small; 300 is the
#: contract's floor for judge/answer calls.
JUDGE_MAX_TOKENS = 400

VALID_VERDICTS = ("correct", "incorrect", "partial")

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*(.*?)```", re.DOTALL)
_CONFIDENCE_RE = re.compile(
    r"""["']?confidence["']?\s*[:=]\s*["']?(-?[0-9]*\.?[0-9]+)""", re.IGNORECASE
)
_VERDICT_FIELD_RE = re.compile(
    r"""["']?verdict["']?\s*[:=]\s*["']?\s*(correct|incorrect|partial|not\s+correct|wrong|right|yes|no)""",
    re.IGNORECASE,
)
#: Negation first: "the answer is not correct" must not match the bare ``correct``.
_NEGATED_CORRECT_RE = re.compile(
    r"\b(?:not|isn'?t|is\s+not|n't)\s+(?:fully\s+|quite\s+|exactly\s+)?correct\b",
    re.IGNORECASE,
)
_INCORRECT_RE = re.compile(r"\bincorrect\b", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\bpartial(?:ly)?\b", re.IGNORECASE)
_CORRECT_RE = re.compile(r"\bcorrect\b", re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    """One graded answer.

    Attributes
    ----------
    label
        ``"correct"``, ``"incorrect"``, ``"partial"``, or ``"unparsed"``.
    correct
        ``True`` / ``False`` for the headline metric, or ``None`` when no verdict could
        be extracted. **``partial`` maps to ``False``** — a partially recovered fact is
        not an answer the agent could act on — but the label keeps it separable so the
        report can show the near-misses without letting them into ``n_correct``.
    confidence
        The judge's self-reported confidence, clamped to ``[0, 1]``, or ``None``.
    raw
        Verbatim judge output (or the error text). Never discarded: this is what makes
        the verdicts re-auditable.
    """

    label: str
    correct: bool | None
    confidence: float | None
    raw: str

    @property
    def parsed(self) -> bool:
        return self.label != "unparsed"


UNPARSED = "unparsed"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_judge_prompt(probe: Probe, answer: str) -> str:
    """Render the grading prompt for one ``(probe, answer)`` pair.

    The answer is passed through verbatim (not summarised, not truncated): a judge
    that cannot see the full candidate cannot fairly call a "no" — and the answerer is
    instructed to answer in one or two sentences, so the prompt stays small.
    """
    return JUDGE_RULES.format(
        question=probe.question.strip(),
        reference_answer=probe.reference_answer.strip(),
        answer=answer.strip() if answer.strip() else "(empty answer)",
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text unchanged."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Unterminated fence: drop the opening marker and let the JSON scan take over.
    return re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text.strip(), count=1).strip()


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` block, respecting strings and escapes.

    ``json.loads`` needs exactly one object and no trailing prose. Models routinely
    emit ``Here is the verdict: {...}. Let me know if...``, so a brace-balanced scan is
    more robust than ``re.search(r"\\{.*\\}", ...)``, which would run to the last brace
    and swallow prose.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalise_label(value: Any) -> str | None:
    """Map a model-emitted verdict token onto our three labels."""
    if not isinstance(value, str):
        return None
    token = value.strip().lower().replace("_", " ").replace("-", " ")
    token = re.sub(r"\s+", " ", token)
    if token in ("correct", "right", "yes", "true"):
        return "correct"
    if token in ("incorrect", "wrong", "no", "false", "not correct"):
        return "incorrect"
    if token.startswith("partial") or token in ("partly correct", "partially correct"):
        return "partial"
    return None


def _clamp_confidence(value: Any) -> float | None:
    """Coerce a confidence to ``[0, 1]``; tolerate percentages and numeric strings.

    Models answer on two conventions: a fraction (``0.9``) or a percentage (``90``).
    Values in ``(1, 100]`` are read as percentages; anything outside that range is
    treated as a malformed number and clamped to the valid interval, because a
    "confidence" of ``350`` is a formatting error, not a 350% belief.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    # Models sometimes answer "95" meaning 95%. Only integer-valued numbers above 1 are
    # read that way: a fractional 2.5 is a malformed fraction, not 2.5%.
    if 1.0 < num <= 100.0 and float(num).is_integer():
        num /= 100.0
    return round(min(1.0, max(0.0, num)), 4)


def _parse_confidence_from_text(text: str) -> float | None:
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return None
    return _clamp_confidence(match.group(1))


def parse_verdict(raw: str) -> Verdict:
    """Turn a raw judge reply into a :class:`Verdict`. Never raises.

    Four-stage ladder (first hit wins):

    1. **fenced / plain JSON** — the contract's expected shape;
    2. **prose + JSON** — a balanced ``{...}`` block anywhere in the text;
    3. **keyword regex** — negation-aware, for replies that ignored the JSON format;
    4. **give up** — ``label="unparsed"``, ``correct=None``.

    Stage 3 is ordered ``incorrect`` → ``partial`` → ``correct`` because the substrings
    overlap ("not correct", "partially correct") and the conservative reading is the
    one that cannot inflate the accuracy of a compressor.
    """
    text = raw or ""

    # 1+2: JSON, fenced or embedded.
    for candidate in (_strip_fences(text), text):
        blob = _first_json_object(candidate)
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        label = _normalise_label(data.get("verdict") or data.get("label") or data.get("result"))
        if label is None:
            # Some models emit {"correct": true} instead of the verdict enum.
            if isinstance(data.get("correct"), bool):
                label = "correct" if data["correct"] else "incorrect"
        if label is not None:
            confidence = _clamp_confidence(data.get("confidence"))
            if confidence is None:
                confidence = _clamp_confidence(data.get("score"))
            return Verdict(label=label, correct=label == "correct",
                           confidence=confidence, raw=text)

    # 3: keyword fallback.
    field = _VERDICT_FIELD_RE.search(text)
    if field:
        label = _normalise_label(field.group(1))
        if label is not None:
            return Verdict(label=label, correct=label == "correct",
                           confidence=_parse_confidence_from_text(text), raw=text)

    if _NEGATED_CORRECT_RE.search(text) or _INCORRECT_RE.search(text):
        return Verdict(label="incorrect", correct=False,
                       confidence=_parse_confidence_from_text(text), raw=text)
    if _PARTIAL_RE.search(text):
        return Verdict(label="partial", correct=False,
                       confidence=_parse_confidence_from_text(text), raw=text)
    if _CORRECT_RE.search(text):
        return Verdict(label="correct", correct=True,
                       confidence=_parse_confidence_from_text(text), raw=text)

    # 4: no verdict. ``None`` keeps it out of the accuracy denominator.
    return Verdict(label=UNPARSED, correct=None, confidence=None, raw=text)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
def judge_verdict(client: Any, probe: Probe, answer: str) -> Verdict:
    """Ask the judge model to grade ``answer`` against ``probe``'s reference fact.

    Uses role ``"judge"`` — never a hard-coded model id — so the independence of the
    judge from the answerer is guaranteed by configuration, not by convention.

    ``temperature=0`` because this is a classification, not a generation: we want the
    same verdict on a re-run (and the disk cache keys on the full prompt, so a cached
    re-run is free). ``max_tokens`` is well above the ~20 tokens the JSON reply needs
    because reasoning models burn budget on hidden tokens and return empty content
    when the ceiling is too low.

    Network / gateway failures propagate (``LLMError``); parse failures do not.
    ``judge_answer`` is the never-raising wrapper for callers that just want a tuple.
    """
    prompt = build_judge_prompt(probe, answer)
    raw = client.chat(
        "judge",
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        tag=f"judge:{probe.probe_id}",
    )
    return parse_verdict(raw or "")


def judge_answer(
    client: Any, probe: Probe, answer: str
) -> tuple[bool | None, str, float | None]:
    """Grade one answer, returning ``(correct, raw_judge_text, confidence)``.

    This is the contract's public surface. It **never raises**: a gateway failure is
    reported as ``(None, "[judge-error] ...", None)`` so a single bad cell cannot abort
    a multi-hundred-call run, and ``None`` (rather than ``False``) keeps an infrastructure
    failure out of the accuracy denominator instead of silently marking the compressor
    wrong.

    ``partial`` returns ``correct=False`` — a partially recovered fact is not usable —
    while ``parse_verdict`` keeps the label available for the report. Callers that need
    the label (``evaluate``, ``cost``) should use :func:`judge_verdict` or
    :func:`verdict_label`.
    """
    try:
        verdict = judge_verdict(client, probe, answer)
    except Exception as exc:  # noqa: BLE001 - one bad cell must not abort the run
        return None, f"[judge-error] {type(exc).__name__}: {exc}", None
    return verdict.correct, verdict.raw, verdict.confidence


def verdict_label(raw: str, correct: bool | None) -> str:
    """Recover the verdict label for a persisted answer row.

    ``Answer`` has no ``partial`` field (and the schema is frozen), so the label is
    re-derived from the retained raw judge text. This keeps ``partial`` countable in
    ``metrics.json`` without a schema change, and it is deterministic: the same raw
    text always yields the same label.

    A row with no verdict stays ``"unparsed"``. If the raw text was not retained (a
    hand-built fixture, or a row written by an older version) but ``correct`` is known,
    the boolean is authoritative — otherwise a perfectly good row would be mislabelled
    as unjudged.
    """
    if correct is None:
        return UNPARSED
    label = parse_verdict(raw or "").label
    if label == UNPARSED:
        return "correct" if correct else "incorrect"
    return label
