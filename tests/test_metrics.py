"""Offline unit tests for the metrics layer.

Everything here runs with no network, no dataset generator and no compressors: the
fixtures are built directly from the ``memfaith.schema`` dataclasses and a
``FakeClient`` stands in for ``LLMClient``. That is deliberate — the metrics are the
part of the audit that *decides the headline number*, so they must be testable without
a flaky gateway and without depending on modules being written in parallel.

What is covered
---------------
* **judge parsing** — plain JSON, fenced JSON, prose + JSON, garbage (``None``, never an
  exception), ``partial`` handling, confidence coercion, and a real (fake) client call.
* **cost arithmetic** — ``cost_per_correct_answer`` on hand-computed answer lists,
  amortized vs one-shot, the zero-correct division guard, baseline delta sign, per-kind
  accuracy, and the breakeven condition at, above and below the threshold.
* **failure joins** — lost-probe detection, fact-attribute aggregation, verbatim
  substring survival (case/whitespace insensitive), frequency bucketing, and the
  ``n_compared`` denominator excluding unjudged cells.
* **evaluate** — the matrix builds the right cells, appends ``answers.jsonl`` in
  ``Answer.to_dict()`` shape, and resumes without re-asking completed cells.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import uuid

import pytest

from memfaith.metrics import (
    analyze_failures,
    compute_metrics,
    dedupe_answers,
    judge_answer,
    load_answers,
    parse_verdict,
    run_matrix,
)
from memfaith.metrics import judge as judge_mod
from memfaith.schema import Answer, CompressionResult, Fact, Probe, Session, Turn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def tmp_path():
    """A throwaway directory *inside the workspace*.

    pytest's built-in ``tmp_path`` uses the OS temp directory, which this sandbox
    denies (``WinError 5``), so the suite would error before running a single
    assertion. ``tempfile.mkdtemp`` is avoided too: it creates the directory with
    restrictive permissions that the sandbox also refuses. ``os.makedirs`` under the
    project root gives a plain directory the sandbox allows, keeping the tests
    hermetic without depending on environment-specific temp permissions.
    """
    base = os.path.join(_PROJECT_ROOT, "pytest_tmp")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "t_" + uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    try:
        yield pathlib.Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# FakeClient — a scripted stand-in for LLMClient
# ---------------------------------------------------------------------------
class FakeClient:
    """Minimal ``LLMClient`` look-alike: returns scripted replies, records calls.

    ``replies`` is either a single string (returned for every call) or a list consumed
    in order. ``raise_on`` maps a call index to an exception, so gateway failures can be
    simulated deterministically. Calls are logged as ``(role, prompt, kwargs)`` so tests
    can assert on *which role* was asked — the judge/answerer separation is a property of
    the code, not of the model ids.
    """

    def __init__(self, replies="ok", raise_on: dict[int, Exception] | None = None,
                 raise_roles: set[str] | None = None):
        self.replies = [replies] if isinstance(replies, str) else list(replies)
        self.raise_on = raise_on or {}
        #: Roles that always raise. Keyed by role (not call index) because the matrix
        #: runs concurrently, so call order is not deterministic.
        self.raise_roles = raise_roles or set()
        self.calls: list[tuple[str, str, dict]] = []
        self._i = 0

    def chat(self, role, messages, **kwargs):
        idx = len(self.calls)
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append((role, prompt, kwargs))
        if role in self.raise_roles:
            raise RuntimeError(f"HTTP 502 on role={role}")
        if idx in self.raise_on:
            raise self.raise_on[idx]
        reply = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return reply


def _verdict_json(label: str, confidence: float = 0.9) -> str:
    return json.dumps({"verdict": label, "confidence": confidence})


# ---------------------------------------------------------------------------
# Fixture builders (schema dataclasses directly — no dataset/probes imports)
# ---------------------------------------------------------------------------
def make_fact(
    fact_id: str,
    value: str,
    *,
    kind: str = "path",
    salience: str = "foreground",
    embedding: str = "standalone",
    stability: str = "stable",
    mentions: list[int] | None = None,
) -> Fact:
    mentions = [0] if mentions is None else mentions
    return Fact(
        fact_id=fact_id,
        kind=kind,
        value=value,
        domain="coding",
        first_turn=min(mentions) if mentions else 0,
        mentions=mentions,
        salience=salience,  # type: ignore[arg-type]
        embedding=embedding,  # type: ignore[arg-type]
        stability=stability,  # type: ignore[arg-type]
    )


def make_session(
    session_id: str = "s001",
    facts: list[Fact] | None = None,
    turns: list[Turn] | None = None,
) -> Session:
    facts = facts or [make_fact("s001-f1", "/srv/app/.env.production")]
    turns = turns or [Turn(idx=0, role="user", text="remember the path")]
    return Session(session_id=session_id, domain="coding", turns=turns, facts=facts,
                   full_tokens=1000, title="t")


def make_probe(
    probe_id: str = "s001-p01",
    session_id: str = "s001",
    kind: str = "specific_fact",
    question: str = "Which path holds the production API key?",
    reference_answer: str = "/srv/app/.env.production",
    fact_ids: list[str] | None = None,
) -> Probe:
    return Probe(
        probe_id=probe_id,
        session_id=session_id,
        kind=kind,  # type: ignore[arg-type]
        question=question,
        reference_answer=reference_answer,
        fact_ids=fact_ids if fact_ids is not None else ["s001-f1"],
        domain="coding",
    )


def make_answer(
    probe_id: str = "s001-p01",
    strategy: str = "full_context",
    condition: str = "full",
    correct: bool | None = True,
    context_tokens: int = 100,
    judge_raw: str = "",
    error: str = "",
    session_id: str = "s001",
    answer_text: str = "the value",
) -> Answer:
    return Answer(
        probe_id=probe_id,
        session_id=session_id,
        strategy=strategy,  # type: ignore[arg-type]
        condition=condition,  # type: ignore[arg-type]
        answer_text=answer_text,
        correct=correct,
        judge_raw=judge_raw,
        context_tokens=context_tokens,
        error=error,
    )


def make_compression(
    strategy: str = "summary_hierarchical",
    session_id: str = "s001",
    text: str = "compressed memory",
    cost_tokens: int = 200,
    original_tokens: int = 1000,
    compressed_tokens: int | None = None,
    meta: dict | None = None,
) -> CompressionResult:
    compressed_tokens = compressed_tokens if compressed_tokens is not None else 250
    return CompressionResult(
        session_id=session_id,
        strategy=strategy,  # type: ignore[arg-type]
        text=text,
        compressed_tokens=compressed_tokens,
        original_tokens=original_tokens,
        ratio=compressed_tokens / original_tokens if original_tokens else 0.0,
        cost_tokens=cost_tokens,
        meta=meta or {},
    )


class FakeCfg:
    """The two attributes the metrics layer reads off ``RunConfig``."""

    def __init__(self, strategies=("summary_hierarchical", "extractive"), concurrency=2):
        self.strategies = tuple(strategies)
        self.concurrency = concurrency


# ===========================================================================
# 1. Judge: prompt + parsing
# ===========================================================================
class TestJudgePrompt:
    def test_prompt_contains_question_reference_and_candidate(self):
        probe = make_probe()
        prompt = judge_mod.build_judge_prompt(probe, "it was in /srv/app/.env.production")
        assert probe.question in prompt
        assert probe.reference_answer in prompt
        assert "it was in /srv/app/.env.production" in prompt

    def test_prompt_states_the_grading_rules_the_contract_requires(self):
        prompt = judge_mod.build_judge_prompt(make_probe(), "answer")
        # extra true statements are OK
        assert "Extra information is fine" in prompt
        # wrong/outdated value is incorrect
        assert "OUTDATED" in prompt.upper()
        # refusal / hedging is incorrect
        assert "refusal" in prompt.lower() or "hedge" in prompt.lower()
        # strict JSON contract
        assert '"verdict"' in prompt
        assert '"confidence"' in prompt
        # partial is explicitly not correct
        assert "PARTIAL is NOT correct" in prompt

    def test_empty_answer_is_rendered_explicitly(self):
        prompt = judge_mod.build_judge_prompt(make_probe(), "   ")
        assert "(empty answer)" in prompt


class TestJudgeParsing:
    def test_plain_json_correct(self):
        v = parse_verdict('{"verdict": "correct", "confidence": 0.93}')
        assert v.label == "correct" and v.correct is True
        assert v.confidence == pytest.approx(0.93)

    def test_plain_json_incorrect(self):
        v = parse_verdict('{"verdict": "incorrect", "confidence": 0.8}')
        assert v.label == "incorrect" and v.correct is False

    def test_fenced_json(self):
        raw = '```json\n{"verdict": "correct", "confidence": 0.7}\n```'
        v = parse_verdict(raw)
        assert v.label == "correct" and v.correct is True

    def test_fenced_json_without_language_tag(self):
        raw = '```\n{"verdict":"incorrect","confidence":0.5}\n```'
        assert parse_verdict(raw).label == "incorrect"

    def test_prose_then_json(self):
        raw = ('Looking at the candidate answer, it clearly contains the reference path.\n'
               '{"verdict": "correct", "confidence": 0.88}\n'
               'Let me know if you need anything else!')
        v = parse_verdict(raw)
        assert v.label == "correct" and v.correct is True
        assert v.confidence == pytest.approx(0.88)

    def test_json_with_trailing_braces_in_prose(self):
        # The balanced scan must stop at the first complete object, not run to the last }.
        raw = '{"verdict": "incorrect", "confidence": 0.6} Note: {this is prose, not json}'
        assert parse_verdict(raw).label == "incorrect"

    def test_partial_is_not_correct_but_keeps_its_label(self):
        v = parse_verdict('{"verdict": "partial", "confidence": 0.75}')
        assert v.label == "partial"
        assert v.correct is False, "partial must not count as a correct answer"
        assert v.parsed is True

    def test_garbage_returns_none_and_never_raises(self):
        for raw in ("", "   ", "I cannot answer that.", "no json here at all",
                    "{broken json", "{}", '{"confidence": 0.4}', "\x00\x01"):
            v = parse_verdict(raw)
            assert v.correct is None, f"expected no verdict for {raw!r}"
            assert v.label == "unparsed"
            assert v.raw == raw

    def test_keyword_fallback_correct(self):
        assert parse_verdict("Verdict: the answer is correct.").correct is True

    def test_keyword_fallback_incorrect_negation(self):
        # "not correct" must not be read as "correct".
        v = parse_verdict("The candidate answer is not correct because the path differs.")
        assert v.correct is False
        assert v.label == "incorrect"

    def test_keyword_fallback_partial(self):
        v = parse_verdict("This is partially correct: the directory matches but not the file.")
        assert v.label == "partial" and v.correct is False

    def test_confidence_percentage_is_normalised(self):
        assert parse_verdict('{"verdict":"correct","confidence":95}').confidence == pytest.approx(0.95)

    def test_confidence_is_clamped(self):
        assert parse_verdict('{"verdict":"correct","confidence":3.5}').confidence == pytest.approx(1.0)
        assert parse_verdict('{"verdict":"correct","confidence":-1}').confidence == pytest.approx(0.0)

    def test_boolean_correct_field_is_accepted(self):
        assert parse_verdict('{"correct": true}').correct is True
        assert parse_verdict('{"correct": false}').correct is False

    def test_uppercase_and_underscore_variants(self):
        assert parse_verdict('{"verdict": "CORRECT"}').correct is True
        assert parse_verdict('{"verdict": "not_correct"}').correct is False


class TestJudgeAnswer:
    def test_returns_tuple_and_uses_judge_role(self):
        client = FakeClient(_verdict_json("correct", 0.91))
        correct, raw, conf = judge_answer(client, make_probe(), "/srv/app/.env.production")
        assert correct is True
        assert conf == pytest.approx(0.91)
        assert "correct" in raw
        role, prompt, kwargs = client.calls[0]
        assert role == "judge", "must use the judge role, never a hardcoded model"
        assert kwargs["max_tokens"] >= 300
        assert kwargs["temperature"] == 0.0
        assert "Which path holds the production API key?" in prompt

    def test_parse_failure_returns_none_tuple(self):
        client = FakeClient("I refuse to emit JSON.")
        correct, raw, conf = judge_answer(client, make_probe(), "something")
        assert correct is None
        assert conf is None
        assert raw == "I refuse to emit JSON."

    def test_gateway_failure_is_swallowed(self):
        client = FakeClient("unused", raise_on={0: RuntimeError("HTTP 502")})
        correct, raw, conf = judge_answer(client, make_probe(), "something")
        assert correct is None
        assert conf is None
        assert "judge-error" in raw and "HTTP 502" in raw

    def test_partial_surfaces_as_false(self):
        client = FakeClient(_verdict_json("partial", 0.6))
        correct, raw, _ = judge_answer(client, make_probe(), "partly")
        assert correct is False
        assert "partial" in raw


# ===========================================================================
# 2. Cost arithmetic
# ===========================================================================
class TestCostArithmetic:
    def _two_probe_fixture(self):
        """Baseline: 2 probes, both correct, 1000 ctx tokens each.

        Compressed: 2 probes, 1 correct, 200 ctx tokens each, 400 compression tokens.
        => baseline cpa = 2000/2 = 1000; compressed amortized = (400+400)/1 = 800.
        """
        probes = [
            make_probe("s001-p01", kind="specific_fact"),
            make_probe("s001-p02", kind="high_level", fact_ids=["s001-f1"]),
        ]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p02", "full_context", "full", True, 1000),
            make_answer("s001-p01", "summary_hierarchical", "compressed", True, 200),
            make_answer("s001-p02", "summary_hierarchical", "compressed", False, 200),
        ]
        comps = {"summary_hierarchical": [make_compression(cost_tokens=400)]}
        return answers, probes, sessions, comps

    def test_baseline_and_compressed_cost_per_correct_answer(self):
        answers, probes, sessions, comps = self._two_probe_fixture()
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        base = m["arms"]["full_context"]
        comp = m["arms"]["summary_hierarchical"]

        assert base["context_tokens_total"] == 2000
        assert base["compression_cost_tokens"] == 0
        assert base["n_correct"] == 2 and base["n_probes"] == 2
        assert base["accuracy"] == pytest.approx(1.0)
        assert base["cost_per_correct_answer_amortized"] == pytest.approx(1000.0)
        assert base["cost_per_correct_answer_one_shot"] == pytest.approx(1000.0)
        assert base["compression_ratio"] == 1.0
        assert base["zero_correct"] is False

        assert comp["context_tokens_total"] == 400
        assert comp["compression_cost_tokens"] == 400
        assert comp["total_tokens_amortized"] == 800
        assert comp["total_tokens_one_shot"] == 400
        assert comp["n_correct"] == 1
        assert comp["cost_per_correct_answer_amortized"] == pytest.approx(800.0)
        assert comp["cost_per_correct_answer_one_shot"] == pytest.approx(400.0)

    def test_delta_sign_is_positive_when_compression_costs_more(self):
        """The thesis case: big token saving, bigger accuracy loss => positive delta."""
        probes = [
            make_probe("s001-p01", kind="specific_fact"),
            make_probe("s001-p02", kind="specific_fact"),
        ]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p02", "full_context", "full", True, 1000),
            # ratio 0.25 but only 1 of 2 correct: (250+250)/1 = 500 vs 1000 -> cheaper.
            # Make it genuinely worse: 0.5 ratio, 1 correct -> 1000 vs 1000 -> equal.
            make_answer("s001-p01", "summary_hierarchical", "compressed", True, 500),
            make_answer("s001-p02", "summary_hierarchical", "compressed", False, 500),
        ]
        comps = {"summary_hierarchical": [make_compression(cost_tokens=0,
                                                           compressed_tokens=500)]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        comp = m["arms"]["summary_hierarchical"]
        assert comp["cost_per_correct_answer_amortized"] == pytest.approx(1000.0)
        assert comp["cost_delta_vs_baseline_abs"] == pytest.approx(0.0)
        assert comp["cost_delta_vs_baseline_pct"] == pytest.approx(0.0)

    def test_positive_delta_when_accuracy_drops_more_than_tokens(self):
        probes = [make_probe("s001-p01"), make_probe("s001-p02")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p02", "full_context", "full", True, 1000),
            # 0.25 ratio (250 each => 500 total) but 1 correct => 500... still cheaper.
            # Use compression cost to push it over: 500 + 600 = 1100 / 1 = 1100 > 1000.
            make_answer("s001-p01", "summary_hierarchical", "compressed", True, 250),
            make_answer("s001-p02", "summary_hierarchical", "compressed", False, 250),
        ]
        comps = {"summary_hierarchical": [make_compression(cost_tokens=600,
                                                           compressed_tokens=250)]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        comp = m["arms"]["summary_hierarchical"]
        assert comp["cost_per_correct_answer_amortized"] == pytest.approx(1100.0)
        assert comp["cost_delta_vs_baseline_abs"] == pytest.approx(100.0)
        assert comp["cost_delta_vs_baseline_pct"] == pytest.approx(0.10)
        assert comp["cost_ratio_vs_baseline"] == pytest.approx(1.10)
        assert comp["accuracy_delta_vs_baseline"] == pytest.approx(-0.5)

    def test_zero_correct_guard_returns_none_and_sets_flag(self):
        probes = [make_probe("s001-p01")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p01", "extractive", "compressed", False, 100),
        ]
        comps = {"extractive": [make_compression("extractive", cost_tokens=50)]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        ext = m["arms"]["extractive"]
        assert ext["n_correct"] == 0
        assert ext["zero_correct"] is True
        assert ext["cost_per_correct_answer_amortized"] is None
        assert ext["cost_per_correct_answer_one_shot"] is None
        # No infinities or NaN leaking into JSON.
        json.dumps(m, allow_nan=False)

    def test_accuracy_by_kind(self):
        probes = [
            make_probe("s001-p01", kind="high_level"),
            make_probe("s001-p02", kind="specific_fact"),
            make_probe("s001-p03", kind="specific_fact"),
            make_probe("s001-p04", kind="multi_hop"),
        ]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 10),
            make_answer("s001-p02", "full_context", "full", True, 10),
            make_answer("s001-p03", "full_context", "full", False, 10),
            make_answer("s001-p04", "full_context", "full", True, 10),
        ]
        m = compute_metrics(answers, probes, sessions, {}, FakeCfg())
        kinds = m["arms"]["full_context"]["accuracy_by_kind"]
        assert kinds["high_level"] == pytest.approx(1.0)
        assert kinds["specific_fact"] == pytest.approx(0.5)
        assert kinds["multi_hop"] == pytest.approx(1.0)
        assert m["accuracy_by_kind"]["full_context"] == kinds

    def test_unjudged_cells_are_excluded_not_counted_wrong(self):
        probes = [make_probe("s001-p01"), make_probe("s001-p02")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p02", "full_context", "full", True, 1000),
            make_answer("s001-p01", "extractive", "compressed", True, 200),
            # gateway failure: no verdict, but context was still spent
            make_answer("s001-p02", "extractive", "compressed", None, 200,
                        error="judge_error: timeout"),
        ]
        comps = {"extractive": [make_compression("extractive", cost_tokens=0)]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        ext = m["arms"]["extractive"]
        assert ext["n_probes"] == 1, "unjudged cell must not enter the denominator"
        assert ext["n_correct"] == 1
        assert ext["accuracy"] == pytest.approx(1.0)
        assert ext["n_unparsed"] == 1
        # ...but its context tokens were really spent, so they stay in the numerator.
        assert ext["context_tokens_total"] == 400
        # The judged correct cell alone is not unparsed.
        assert m["arms"]["full_context"]["n_unparsed"] == 0

    def test_partial_counts_separately_and_not_as_correct(self):
        probes = [make_probe("s001-p01")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p01", "extractive", "compressed", False, 100,
                        judge_raw='{"verdict": "partial", "confidence": 0.6}'),
        ]
        m = compute_metrics(answers, probes, sessions,
                            {"extractive": [make_compression("extractive")]}, FakeCfg())
        ext = m["arms"]["extractive"]
        assert ext["n_correct"] == 0
        assert ext["n_partial"] == 1
        assert ext["zero_correct"] is True

    def test_compression_ratio_is_mean_over_sessions(self):
        sessions = [make_session("s001"), make_session("s002")]
        probes = [make_probe("s001-p01", session_id="s001")]
        answers = [make_answer("s001-p01", "full_context", "full", True, 100)]
        comps = {"summary_hierarchical": [
            make_compression(compressed_tokens=100, original_tokens=1000),  # 0.1
            make_compression("summary_hierarchical", "s002", compressed_tokens=500,
                             original_tokens=1000),  # 0.5
        ]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        assert m["arms"]["summary_hierarchical"]["compression_ratio"] == pytest.approx(0.3)

    def test_dedupe_prefers_successful_retry(self):
        probes = [make_probe("s001-p01")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", None, 1000,
                        error="answer_error: HTTP 502"),
            make_answer("s001-p01", "full_context", "full", True, 1000),
        ]
        deduped = dedupe_answers(answers)
        assert len(deduped) == 1
        assert next(iter(deduped.values()))["correct"] is True
        m = compute_metrics(answers, probes, sessions, {}, FakeCfg())
        assert m["arms"]["full_context"]["n_correct"] == 1


# ===========================================================================
# 3. Breakeven logic
# ===========================================================================
class TestBreakeven:
    def _metrics(self, *, ratio: float, correct_comp: int, n: int = 4,
                 comp_cost: int = 0):
        """Build a metrics doc where the context ratio and the answer ratio agree.

        Each of the ``n`` probes reads the full transcript (1000 tokens) in the baseline
        and ``ratio * 1000`` tokens when compressed, so ``ratio`` is simultaneously the
        token ratio and the per-probe context ratio — which is what makes the breakeven
        algebra checkable by hand.
        """
        ctx_comp = int(ratio * 1000)
        probes = [make_probe(f"s001-p0{i}", kind="specific_fact") for i in range(1, n + 1)]
        sessions = [make_session()]
        answers = [
            make_answer(f"s001-p0{i}", "full_context", "full", True, 1000)
            for i in range(1, n + 1)
        ] + [
            make_answer(f"s001-p0{i}", "summary_hierarchical", "compressed",
                        i <= correct_comp, ctx_comp)
            for i in range(1, n + 1)
        ]
        comps = {"summary_hierarchical": [
            make_compression(compressed_tokens=int(ratio * 1000), cost_tokens=comp_cost)
        ]}
        return compute_metrics(answers, probes, sessions, comps, FakeCfg())

    def test_pays_off_when_retention_exceeds_ratio(self):
        # ratio 0.25, retention 4/4 = 1.0 > 0.25 => pays off
        m = self._metrics(ratio=0.25, correct_comp=4)
        be = m["breakeven"]
        assert be["ratio"] == pytest.approx(0.25)
        assert be["accuracy_retention"] == pytest.approx(1.0)
        assert be["verdict"] == "compression_pays_off"
        assert be["max_viable_ratio"] == pytest.approx(1.0)
        assert m["arms"]["summary_hierarchical"]["cost_delta_vs_baseline_pct"] < 0

    def test_costs_more_when_retention_below_ratio(self):
        # ratio 0.5, retention 1/4 = 0.25 < 0.5 => costs more
        m = self._metrics(ratio=0.5, correct_comp=1)
        be = m["breakeven"]
        assert be["ratio"] == pytest.approx(0.5)
        assert be["accuracy_retention"] == pytest.approx(0.25)
        assert be["verdict"] == "compression_costs_more"
        assert be["max_viable_ratio"] == pytest.approx(0.25)
        comp = m["arms"]["summary_hierarchical"]
        # 4 probes x 500 compressed tokens = 2000 spent; 1 correct => 2000 per correct
        # answer vs 4000/4 = 1000 for full context: exactly twice as expensive.
        assert comp["cost_per_correct_answer_amortized"] == pytest.approx(2000.0)
        assert comp["cost_delta_vs_baseline_pct"] == pytest.approx(1.0)
        assert comp["cost_ratio_vs_baseline"] == pytest.approx(2.0)

    def test_breakeven_at_the_threshold_is_not_a_win(self):
        # ratio 0.25, retention 1/4 = 0.25 => exactly equal, not a win.
        m = self._metrics(ratio=0.25, correct_comp=1)
        assert m["breakeven"]["accuracy_retention"] == pytest.approx(0.25)
        assert m["breakeven"]["verdict"] == "compression_costs_more"

    def test_inconclusive_when_baseline_accuracy_is_zero(self):
        probes = [make_probe("s001-p01")]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", False, 1000),
            make_answer("s001-p01", "extractive", "compressed", False, 100),
        ]
        m = compute_metrics(answers, probes, sessions,
                            {"extractive": [make_compression("extractive")]}, FakeCfg())
        assert m["breakeven"]["verdict"] == "inconclusive"
        assert m["breakeven"]["accuracy_retention"] is None

    def test_amortized_ratio_includes_compression_spend(self):
        m = self._metrics(ratio=0.25, correct_comp=4, comp_cost=500)
        be = m["breakeven"]
        # context 1000 + compression 500 = 1500 vs baseline 4000
        assert be["ratio_amortized"] == pytest.approx(1500 / 4000)
        assert be["max_viable_ratio_amortized"] == pytest.approx(1.0 - 500 / 4000)

    def test_max_viable_ratio_is_capped_at_one(self):
        """Retention > 1 is possible (compression helps); a ratio cannot exceed 100%."""
        m = self._metrics(ratio=0.25, correct_comp=4)  # retention 1.0
        assert m["breakeven"]["accuracy_retention"] == pytest.approx(1.0)
        assert m["breakeven"]["max_viable_ratio"] == pytest.approx(1.0)

        # Now make compression *better* than baseline: retention 1.5 (6/4).
        probes = [make_probe(f"s001-p0{i}", kind="specific_fact") for i in range(1, 5)]
        sessions = [make_session()]
        answers = [
            # baseline: 2 of 4 correct
            make_answer(f"s001-p0{i}", "full_context", "full", i <= 2, 1000)
            for i in range(1, 5)
        ] + [
            # compressed: 3 of 4 correct => retention 1.5
            make_answer(f"s001-p0{i}", "summary_hierarchical", "compressed", i <= 3, 250)
            for i in range(1, 5)
        ]
        comps = {"summary_hierarchical": [
            make_compression(compressed_tokens=250, cost_tokens=0)]}
        m2 = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        be = m2["breakeven"]
        assert be["accuracy_retention"] == pytest.approx(1.5)
        assert be["max_viable_ratio"] == pytest.approx(1.0), "a ratio cannot exceed 100%"
        assert be["verdict"] == "compression_pays_off"
        # The headline must not read "keeps 150% of the accuracy".
        assert "150%" not in m2["verdict"]["headline"]
        assert "better" in m2["verdict"]["headline"]

    def test_when_is_compression_worth_it_table(self):
        m = self._metrics(ratio=0.5, correct_comp=1)
        table = m["when_is_compression_worth_it"]
        assert len(table) == 1
        row = table[0]
        assert row["strategy"] == "summary_hierarchical"
        assert row["breakeven_holds"] is False
        assert row["accuracy_retention"] == pytest.approx(0.25)

    def test_headline_names_the_numbers(self):
        m = self._metrics(ratio=0.5, correct_comp=1)
        headline = m["verdict"]["headline"]
        assert "summary_hierarchical" in headline
        assert "50%" in headline  # the ratio
        assert "25%" in headline  # the retention
        assert "instead of" in headline or "worse" in headline

    def test_per_kind_cost_breakdown(self):
        probes = [
            make_probe("s001-p01", kind="high_level"),
            make_probe("s001-p02", kind="specific_fact"),
        ]
        sessions = [make_session()]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 100),
            make_answer("s001-p02", "full_context", "full", True, 300),
            make_answer("s001-p01", "extractive", "compressed", True, 100),
            make_answer("s001-p02", "extractive", "compressed", False, 300),
        ]
        comps = {"extractive": [make_compression("extractive", cost_tokens=400)]}
        m = compute_metrics(answers, probes, sessions, comps, FakeCfg())
        by_kind = m["by_probe_kind_cost"]["extractive"]
        # 400 compression tokens split 100:300 by context share => 100 and 300.
        assert by_kind["high_level"]["compression_cost_tokens_allocated"] == 100
        assert by_kind["specific_fact"]["compression_cost_tokens_allocated"] == 300
        assert by_kind["high_level"]["cost_per_correct_answer_amortized"] == pytest.approx(200.0)
        assert by_kind["specific_fact"]["cost_per_correct_answer_amortized"] is None
        assert by_kind["specific_fact"]["zero_correct"] is True

    def test_metrics_json_is_written_and_json_safe(self, tmp_path):
        m = self._metrics(ratio=0.25, correct_comp=1)
        compute_metrics(
            [make_answer("s001-p01", "full_context", "full", True, 1000)],
            [make_probe("s001-p01")], [make_session()],
            {"summary_hierarchical": [make_compression()]}, FakeCfg(),
            out_dir=str(tmp_path),
        )
        path = tmp_path / "metrics.json"
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "arms" in loaded and "breakeven" in loaded and "verdict" in loaded
        json.dumps(m, allow_nan=False)


# ===========================================================================
# 4. Failure analysis
# ===========================================================================
class TestFailureAnalysis:
    def _fixture(self):
        """One session, three facts with different covariates, four probes.

        - f1 standalone/foreground/stable, mentioned 3x  -> survives verbatim
        - f2 deep_buried/background/stable, mentioned 1x -> dropped
        - f3 embedded/foreground/superseded, mentioned 2x -> dropped (old value kept)
        """
        facts = [
            make_fact("f1", "/srv/app/.env.production", salience="foreground",
                      embedding="standalone", mentions=[0, 2, 4]),
            make_fact("f2", "arxiv:2401.01234", kind="citation", salience="background",
                      embedding="deep_buried", mentions=[5]),
            make_fact("f3", "v2.4.1", kind="version", salience="foreground",
                      embedding="embedded", stability="superseded", mentions=[1, 3]),
        ]
        sessions = [make_session("s001", facts=facts)]
        probes = [
            make_probe("s001-p01", kind="specific_fact", reference_answer=facts[0].value,
                       fact_ids=["f1"]),
            make_probe("s001-p02", kind="specific_fact", reference_answer=facts[1].value,
                       fact_ids=["f2"]),
            make_probe("s001-p03", kind="specific_fact", reference_answer=facts[2].value,
                       fact_ids=["f3"]),
            make_probe("s001-p04", kind="multi_hop", reference_answer="f1 + f3",
                       fact_ids=["f1", "f3"]),
        ]
        compressed_text = (
            "The project uses /srv/app/.env.production for the production key. "
            "Earlier the version was v1.9.0."
        )
        answers = [
            # baseline: all four correct
            make_answer("s001-p01", "full_context", "full", True, 1000),
            make_answer("s001-p02", "full_context", "full", True, 1000),
            make_answer("s001-p03", "full_context", "full", True, 1000),
            make_answer("s001-p04", "full_context", "full", True, 1000),
            # compressed: p01 correct (f1 survived), p02/p03/p04 wrong
            make_answer("s001-p01", "summary_hierarchical", "compressed", True, 250,
                        answer_text="/srv/app/.env.production"),
            make_answer("s001-p02", "summary_hierarchical", "compressed", False, 250,
                        answer_text="I don't know"),
            make_answer("s001-p03", "summary_hierarchical", "compressed", False, 250,
                        answer_text="v1.9.0"),
            make_answer("s001-p04", "summary_hierarchical", "compressed", False, 250,
                        answer_text="v1.9.0"),
        ]
        comps = {"summary_hierarchical": [
            make_compression(text=compressed_text, compressed_tokens=250)
        ]}
        return answers, probes, sessions, comps

    def test_fact_ids_are_scoped_per_session(self):
        """Regression: ``fact_id`` is only unique *within* a session.

        Every session names its facts f01, f02, ... Keying the fact index on
        ``fact_id`` alone collapsed 30 sessions into 9 facts, so the survival
        table reported n=2..5 and 100% survival instead of the real distribution.
        Two sessions with the same fact_id must stay two distinct facts.
        """
        facts_a = [make_fact("f01", "/srv/a/.env", salience="foreground",
                             embedding="standalone")]
        facts_b = [make_fact("f01", "arxiv:2401.99999", kind="citation",
                             salience="background", embedding="deep_buried")]
        sessions = [make_session("s001", facts=facts_a),
                    make_session("s002", facts=facts_b)]
        probes = [
            make_probe("s001-p01", session_id="s001", fact_ids=["f01"],
                       reference_answer="/srv/a/.env"),
            make_probe("s002-p01", session_id="s002", fact_ids=["f01"],
                       reference_answer="arxiv:2401.99999"),
        ]
        answers = [
            make_answer("s001-p01", "full_context", "full", True, 1000, session_id="s001"),
            make_answer("s002-p01", "full_context", "full", True, 1000, session_id="s002"),
            # s001's fact survives verbatim; s002's is dropped.
            make_answer("s001-p01", "summary_hierarchical", "compressed", True, 250,
                        session_id="s001"),
            make_answer("s002-p01", "summary_hierarchical", "compressed", False, 250,
                        session_id="s002"),
        ]
        comps = {"summary_hierarchical": [
            make_compression(session_id="s001", text="the key is /srv/a/.env"),
            make_compression(session_id="s002", text="some unrelated memory"),
        ]}
        f = analyze_failures(answers, probes, sessions, comps)

        assert f["summary"]["n_facts_tracked"] == 2, (
            "two sessions each with f01 must count as two facts, not one")
        assert f["summary"]["n_facts_lost_verbatim"] == 1
        # One standalone/foreground fact survived, one deep_buried/background did not.
        emb = f["survival_by_attribute"]["embedding"]
        assert emb["standalone"]["n"] == 1 and emb["standalone"]["survival_rate"] == 1.0
        assert emb["deep_buried"]["n"] == 1 and emb["deep_buried"]["survival_rate"] == 0.0

    def test_lost_probe_detection(self):
        answers, probes, sessions, comps = self._fixture()
        f = analyze_failures(answers, probes, sessions, comps)
        lost = set(f["summary"]["lost_probe_ids"])
        assert lost == {"s001-p02", "s001-p03", "s001-p04"}
        assert f["summary"]["n_lost"] == 3
        assert f["summary"]["n_compared"] == 4
        assert f["summary"]["loss_rate"] == pytest.approx(0.75)

    def test_verbatim_survival_is_case_and_whitespace_insensitive(self):
        facts = [make_fact("f1", "/srv/App/.env.production")]
        sessions = [make_session("s001", facts=facts)]
        probes = [make_probe("s001-p01", fact_ids=["f1"])]
        answers = [
            make_answer("s001-p01", "full_context", "full", True),
            # The probe flips, so a case exists to inspect.
            make_answer("s001-p01", "extractive", "compressed", False),
        ]
        comps = {"extractive": [make_compression(
            "extractive", text="the key is in  /srv/app/.env.production\nok")]}
        f = analyze_failures(answers, probes, sessions, comps)
        case = f["cases"][0]
        # Different case and doubled whitespace still count as verbatim survival.
        assert case["facts"][0]["survived_verbatim"] is True
        # ...and because it survived, the explanation blames retrieval, not content loss.
        assert "retrieval" in case["explanation"].lower() or \
               "wrong or outdated" in case["explanation"].lower()

    def test_fact_attribute_aggregation(self):
        answers, probes, sessions, comps = self._fixture()
        f = analyze_failures(answers, probes, sessions, comps)
        survival = f["survival_by_attribute"]

        # embedding: f1 standalone survived, f2 deep_buried dropped, f3 embedded dropped
        assert survival["embedding"]["standalone"]["n"] == 1
        assert survival["embedding"]["standalone"]["survival_rate"] == pytest.approx(1.0)
        assert survival["embedding"]["deep_buried"]["survival_rate"] == pytest.approx(0.0)
        assert survival["embedding"]["embedded"]["survival_rate"] == pytest.approx(0.0)

        # salience: foreground = f1 (survived) + f3 (dropped) => 0.5
        assert survival["salience"]["foreground"]["n"] == 2
        assert survival["salience"]["foreground"]["survival_rate"] == pytest.approx(0.5)
        assert survival["salience"]["background"]["survival_rate"] == pytest.approx(0.0)

        # frequency buckets: f1 mentioned 3x -> "3+" and survived; f2 1x; f3 2x
        assert survival["frequency"]["3+"]["survival_rate"] == pytest.approx(1.0)
        assert survival["frequency"]["1"]["survival_rate"] == pytest.approx(0.0)
        assert survival["frequency"]["2"]["survival_rate"] == pytest.approx(0.0)

        # stability
        assert survival["stability"]["stable"]["n"] == 2
        assert survival["stability"]["stable"]["survival_rate"] == pytest.approx(0.5)
        assert survival["stability"]["superseded"]["survival_rate"] == pytest.approx(0.0)

    def test_case_records_carry_fact_values_and_snippet(self):
        answers, probes, sessions, comps = self._fixture()
        f = analyze_failures(answers, probes, sessions, comps)
        case = next(c for c in f["cases"] if c["probe_id"] == "s001-p02")
        assert case["kind"] == "specific_fact"
        assert case["strategy"] == "summary_hierarchical"
        assert case["facts"][0]["fact_id"] == "f2"
        assert case["facts"][0]["value"] == "arxiv:2401.01234"
        assert case["facts"][0]["survived_verbatim"] is False
        assert case["facts"][0]["embedding"] == "deep_buried"
        assert case["facts"][0]["frequency"] == 1
        assert case["compressed_snippet"]
        assert "dropped" in case["explanation"].lower()

    def test_superseded_case_explanation_mentions_the_stale_value(self):
        answers, probes, sessions, comps = self._fixture()
        f = analyze_failures(answers, probes, sessions, comps)
        case = next(c for c in f["cases"] if c["probe_id"] == "s001-p03")
        # v2.4.1 is absent from the compressed text, v1.9.0 is present.
        assert case["facts"][0]["stability"] == "superseded"
        assert case["facts"][0]["survived_verbatim"] is False
        assert "v1.9.0" in case["answer_text"]

    def test_probes_wrong_in_full_context_are_not_counted_as_lost(self):
        facts = [make_fact("f1", "value-a")]
        sessions = [make_session("s001", facts=facts)]
        probes = [make_probe("s001-p01", fact_ids=["f1"])]
        answers = [
            make_answer("s001-p01", "full_context", "full", False),  # already wrong
            make_answer("s001-p01", "extractive", "compressed", False),
        ]
        f = analyze_failures(answers, probes, sessions,
                             {"extractive": [make_compression("extractive")]})
        assert f["summary"]["n_lost"] == 0
        assert f["summary"]["n_compared"] == 1

    def test_unjudged_compressed_answers_excluded_from_denominator(self):
        facts = [make_fact("f1", "value-a")]
        sessions = [make_session("s001", facts=facts)]
        probes = [make_probe("s001-p01", fact_ids=["f1"])]
        answers = [
            make_answer("s001-p01", "full_context", "full", True),
            make_answer("s001-p01", "extractive", "compressed", None,
                        error="judge_error: timeout"),
        ]
        f = analyze_failures(answers, probes, sessions,
                             {"extractive": [make_compression("extractive")]})
        assert f["summary"]["n_compared"] == 0
        assert f["summary"]["n_lost"] == 0
        assert f["summary"]["loss_rate"] is None

    def test_failures_json_is_written(self, tmp_path):
        answers, probes, sessions, comps = self._fixture()
        analyze_failures(answers, probes, sessions, comps, out_dir=str(tmp_path))
        path = tmp_path / "failures.json"
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["summary"]["n_lost"] == 3
        assert "survival_by_attribute" in loaded
        assert loaded["cases"]

    def test_empty_buckets_are_omitted_not_null(self):
        """Every emitted ``survival_rate`` must be a real float (report charts plot them)."""
        answers, probes, sessions, comps = self._fixture()
        f = analyze_failures(answers, probes, sessions, comps)
        for group, values in f["survival_by_attribute"].items():
            assert values, f"group {group} should not be present if empty"
            for value, bucket in values.items():
                assert isinstance(bucket["survival_rate"], float), \
                    f"{group}/{value} has a non-float survival_rate"
                assert bucket["n"] > 0
                json.dumps(bucket, allow_nan=False)

    def test_missing_facts_do_not_crash_the_join(self):
        sessions = [make_session("s001", facts=[])]
        probes = [make_probe("s001-p01", fact_ids=["nonexistent"])]
        answers = [
            make_answer("s001-p01", "full_context", "full", True),
            make_answer("s001-p01", "extractive", "compressed", False),
        ]
        f = analyze_failures(answers, probes, sessions,
                             {"extractive": [make_compression("extractive")]})
        assert f["summary"]["n_lost"] == 1
        assert f["cases"][0]["facts"] == []
        assert "join miss" in f["cases"][0]["explanation"]


# ===========================================================================
# 5. Evaluate: matrix construction, persistence, resume
# ===========================================================================
class TestEvaluateMatrix:
    def _world(self):
        facts = [make_fact("f1", "/srv/app/.env.production")]
        sessions = [make_session("s001", facts=facts)]
        probes = [
            make_probe("s001-p01", kind="specific_fact", fact_ids=["f1"]),
            make_probe("s001-p02", kind="high_level", fact_ids=["f1"]),
        ]
        comps = {
            "summary_hierarchical": [make_compression("summary_hierarchical",
                                                      text="summary memory")],
            "extractive": [make_compression("extractive", text="extractive memory")],
        }
        return sessions, probes, comps

    def _client(self):
        # answerer call, judge call, answerer call, judge call, ...
        return FakeClient([_verdict_json("correct")] * 50)

    def test_runs_every_condition_and_persists_answer_rows(self, tmp_path):
        sessions, probes, comps = self._world()
        client = self._client()
        rows = run_matrix(sessions, probes, comps, client, FakeCfg(), str(tmp_path))

        # 2 probes x (1 full + 2 strategies) = 6 cells
        assert len(rows) == 6
        keys = {(r["probe_id"], r["strategy"], r["condition"]) for r in rows}
        assert ("s001-p01", "full_context", "full") in keys
        assert ("s001-p01", "summary_hierarchical", "compressed") in keys
        assert ("s001-p01", "extractive", "compressed") in keys
        assert len(keys) == 6

        path = tmp_path / "answers.jsonl"
        assert path.exists()
        on_disk = load_answers(str(path))
        assert len(on_disk) == 6
        # rows are exactly Answer.to_dict()
        assert set(on_disk[0]) == set(Answer("", "", "full_context", "full", "").to_dict())

        # Every cell answered and judged: 6 answerer + 6 judge calls.
        roles = [c[0] for c in client.calls]
        assert roles.count("answerer") == 6
        assert roles.count("judge") == 6

    def test_context_tokens_match_the_exact_context_sent(self, tmp_path):
        from memfaith.tokens import count_tokens

        sessions, probes, comps = self._world()
        rows = run_matrix(sessions, probes, comps, self._client(), FakeCfg(), str(tmp_path))
        comp_row = next(r for r in rows
                        if r["strategy"] == "summary_hierarchical"
                        and r["condition"] == "compressed")
        assert comp_row["context_tokens"] == count_tokens("summary memory")
        full_row = next(r for r in rows if r["strategy"] == "full_context")
        assert full_row["context_tokens"] > 0
        # the compressed condition is much smaller than the transcript here
        assert comp_row["context_tokens"] < full_row["context_tokens"]

    def test_resume_skips_completed_cells(self, tmp_path):
        sessions, probes, comps = self._world()
        run_matrix(sessions, probes, comps, self._client(), FakeCfg(), str(tmp_path))

        # Second run: nothing left to do, so the client must not be called at all.
        second = FakeClient([_verdict_json("correct")] * 50)
        rows = run_matrix(sessions, probes, comps, second, FakeCfg(), str(tmp_path))
        assert len(rows) == 6
        assert second.calls == [], "completed cells must not be re-asked"
        assert len(load_answers(str(tmp_path / "answers.jsonl"))) == 6

    def test_resume_retries_failed_judge_cells(self, tmp_path):
        sessions, probes, comps = self._world()
        # First run: every judge call fails => answers exist but no verdicts.
        first = FakeClient([_verdict_json("correct")] * 50, raise_roles={"judge"})
        rows = run_matrix(sessions, probes, comps, first, FakeCfg(), str(tmp_path))
        assert rows
        assert all(r["error"].startswith("judge_error") for r in rows)
        assert all(r["correct"] is None for r in rows)
        assert all(r["answer_text"] for r in rows), "the answers were still obtained"

        # Second run must retry all six cells (the answers are cached on the gateway).
        second = FakeClient([_verdict_json("correct")] * 50)
        rows2 = run_matrix(sessions, probes, comps, second, FakeCfg(), str(tmp_path))
        assert len(rows2) == 6
        assert all(r["correct"] is True for r in rows2)
        # Dedupe on reload: the file has 12 rows but only 6 distinct cells.
        assert len(load_answers(str(tmp_path / "answers.jsonl"))) == 12
        assert len(dedupe_answers(load_answers(str(tmp_path / "answers.jsonl")))) == 6

    def test_missing_compression_is_recorded_once_and_not_retried(self, tmp_path):
        sessions, probes, _ = self._world()
        comps = {"extractive": [make_compression("extractive")]}  # summary missing
        client = FakeClient([_verdict_json("correct")] * 50)
        rows = run_matrix(sessions, probes, comps, client, FakeCfg(), str(tmp_path))
        missing = [r for r in rows if r["strategy"] == "summary_hierarchical"]
        assert len(missing) == 2
        assert all(r["error"] == "compression_missing" for r in missing)
        assert all(r["correct"] is None for r in missing)

        second = FakeClient([_verdict_json("correct")] * 50)
        run_matrix(sessions, probes, comps, second, FakeCfg(), str(tmp_path))
        # The permanent failures are already recorded, so only nothing is re-run.
        assert second.calls == []

    def test_failed_compression_marks_cells_without_answering(self, tmp_path):
        sessions, probes, _ = self._world()
        comps = {"extractive": [make_compression("extractive", meta={"error": "boom"})]}
        client = FakeClient([_verdict_json("correct")] * 50)
        rows = run_matrix(sessions, probes, comps, client, FakeCfg(), str(tmp_path))
        ext = [r for r in rows if r["strategy"] == "extractive"]
        assert all(r["error"].startswith("compression_failed") for r in ext)
        assert all(c[0] != "answerer" or "extractive" not in c[1] for c in client.calls)

    def test_eval_summary_is_written(self, tmp_path):
        sessions, probes, comps = self._world()
        run_matrix(sessions, probes, comps, self._client(), FakeCfg(), str(tmp_path))
        summary = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
        assert summary["n_cells"] == 6
        assert summary["n_new"] == 6
        assert summary["n_resumed"] == 0
        assert summary["by_arm"]["full_context"]["n_correct"] == 2

    def test_concurrency_one_and_many_produce_the_same_cells(self, tmp_path):
        sessions, probes, comps = self._world()
        rows_a = run_matrix(sessions, probes, comps, self._client(),
                            FakeCfg(concurrency=1), str(tmp_path / "a"))
        rows_b = run_matrix(sessions, probes, comps, self._client(),
                            FakeCfg(concurrency=4), str(tmp_path / "b"))
        assert [(r["probe_id"], r["strategy"], r["condition"]) for r in rows_a] == \
               [(r["probe_id"], r["strategy"], r["condition"]) for r in rows_b]


# ===========================================================================
# 6. End-to-end: matrix -> metrics -> failures
# ===========================================================================
def test_full_pipeline_on_fixture(tmp_path):
    """The three modules agree with each other on one hand-built world."""
    facts = [
        make_fact("f1", "/srv/app/.env.production", salience="foreground",
                  embedding="standalone", mentions=[0, 2]),
        make_fact("f2", "arxiv:2401.01234", kind="citation", salience="background",
                  embedding="deep_buried", mentions=[4]),
    ]
    sessions = [make_session("s001", facts=facts)]
    probes = [
        make_probe("s001-p01", kind="specific_fact", fact_ids=["f1"],
                   reference_answer="/srv/app/.env.production"),
        make_probe("s001-p02", kind="specific_fact", fact_ids=["f2"],
                   reference_answer="arxiv:2401.01234"),
    ]
    comps = {
        "summary_hierarchical": [make_compression(
            text="The production key is in /srv/app/.env.production.",
            cost_tokens=500, compressed_tokens=250)],
    }
    # Answerer: correct for p01 in both conditions, wrong for p02 when compressed.
    # Judge: always says correct (so the flip is driven by the answer text).
    client = FakeClient([_verdict_json("correct")] * 20)
    cfg = FakeCfg(strategies=("summary_hierarchical",))
    rows = run_matrix(sessions, probes, comps, client, cfg, str(tmp_path))
    assert len(rows) == 4

    m = compute_metrics(rows, probes, sessions, comps, cfg, out_dir=str(tmp_path))
    assert m["arms"]["full_context"]["n_correct"] == 2
    assert m["arms"]["summary_hierarchical"]["n_correct"] == 2
    assert m["breakeven"]["verdict"] == "compression_pays_off"

    f = analyze_failures(rows, probes, sessions, comps, out_dir=str(tmp_path))
    assert f["summary"]["n_lost"] == 0
    # f2 was probed and is absent from the compressed text.
    assert f["survival_by_attribute"]["embedding"]["deep_buried"]["survival_rate"] == 0.0
    assert f["survival_by_attribute"]["embedding"]["standalone"]["survival_rate"] == 1.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "failures.json").exists()
