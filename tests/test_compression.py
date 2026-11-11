"""Offline tests for the two compression strategies.

Everything here runs with no network and no model download. The ``FakeClient``
mimics the slice of ``LLMClient`` that ``compress()`` uses — ``.chat()`` plus a
``.ledger`` with ``.records`` / ``.totals()`` — so the tests can assert on exact
cost accounting and on the two failure modes the real gateway produces:

* a transient chunk failure (must fall back, not crash), and
* a summarizer that ignores the token budget (must be hard-fitted).

The fixture session is built inline from ``memfaith.schema`` primitives rather
than from ``memfaith.dataset`` so this file has no dependency on the dataset
module another teammate is writing in parallel.
"""

from __future__ import annotations

import re

import pytest

from memfaith.compression import STRATEGIES, get_compressor
from memfaith.compression.base import (
    ledger_delta,
    render_transcript,
    split_by_token_budget,
    truncate_to_token_budget,
)
from memfaith.compression.extractive import ExtractiveCompressor, split_sentences
from memfaith.compression.summary import (
    HierarchicalSummaryCompressor,
    chunk_transcript,
)
from memfaith.llm import Ledger
from memfaith.schema import Fact, Session, Turn
from memfaith.tokens import count_tokens

# ---------------------------------------------------------------------------
# FakeClient
# ---------------------------------------------------------------------------
class FakeClient:
    """Minimal stand-in for ``memfaith.llm.LLMClient``.

    ``chat`` records a ledger entry exactly like the real client (role, tag,
    prompt/completion token counts, cached flag) and returns canned text chosen
    by a responder callable. The token counts are derived from the *actual*
    prompt so cost assertions are meaningful rather than magic numbers.
    """

    def __init__(self, responder=None, completion_tokens: int = 240, models: dict | None = None):
        self.ledger = Ledger()
        self.models = models or {"summarizer": "fake/summarizer-1"}
        self.calls: list[dict] = []
        self.completion_tokens = completion_tokens
        self.responder = responder or self.default_responder

    # -- LLMClient surface --------------------------------------------------
    def chat(self, role: str, messages: list[dict], **kw) -> str:
        prompt_tokens = sum(count_tokens(str(m.get("content", ""))) + 4 for m in messages)
        text = self.responder(role, messages, kw)
        if isinstance(text, Exception):
            self.ledger.add(role=role, tag=kw.get("tag", ""), model=self.models.get(role, "fake"),
                            ok=False, cached=False, prompt_tokens=0, completion_tokens=0,
                            total_tokens=0, latency=0.0, error=str(text))
            raise text
        completion_tokens = count_tokens(text)
        self.ledger.add(role=role, tag=kw.get("tag", ""), model=self.models.get(role, "fake"),
                        ok=True, cached=False, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens, latency=0.01)
        self.calls.append({"role": role, "messages": messages, **kw})
        return text

    def flush(self) -> None:  # pragma: no cover - interface parity
        pass

    def usage(self) -> dict:
        return self.ledger.totals()

    # -- responders ---------------------------------------------------------
    def default_responder(self, role: str, messages: list[dict], kw: dict) -> str:
        """Return a plausible bullet summary that respects nothing about budget."""
        body = messages[-1]["content"]
        m = re.search(r"--- BEGIN SLICE ---\n(.*?)--- END SLICE ---", body, re.DOTALL)
        source = m.group(1) if m else body
        lines = [ln.strip() for ln in source.splitlines() if ln.strip()][:6]
        return "\n".join(f"- {ln}" for ln in lines)

    def chunk_aware_responder(self, role: str, messages: list[dict], kw: dict) -> str:
        """Summarize by keeping the highest-value lines; used for ratio tests."""
        body = messages[-1]["content"]
        keep = []
        for line in body.splitlines():
            if any(tok in line for tok in ("/srv/", "==", "arxiv", "internal", "remember", "v2.")):
                keep.append(line.strip())
        return "\n".join(f"- {ln}" for ln in keep[:8]) or "- (nothing salient)"


class FlakyClient(FakeClient):
    """Fails the first ``n_failures`` summarizer calls, then behaves normally."""

    def __init__(self, n_failures: int = 1, **kw):
        super().__init__(**kw)
        self.n_failures = n_failures
        self._seen = 0

    def chat(self, role: str, messages: list[dict], **kw) -> str:
        self._seen += 1
        if self._seen <= self.n_failures:
            prompt_tokens = sum(count_tokens(str(m.get("content", ""))) + 4 for m in messages)
            self.ledger.add(role=role, tag=kw.get("tag", ""), model="fake", ok=False,
                            cached=False, prompt_tokens=0, completion_tokens=0,
                            total_tokens=0, latency=0.1, error="HTTP 400: simulated")
            raise RuntimeError("simulated gateway 400")
        return super().chat(role, messages, **kw)


class ExplodingClient(FakeClient):
    """Every call fails: the compressor must still return a valid result."""

    def chat(self, role: str, messages: list[dict], **kw) -> str:
        self.ledger.add(role=role, tag=kw.get("tag", ""), model="fake", ok=False,
                        cached=False, prompt_tokens=0, completion_tokens=0,
                        total_tokens=0, latency=0.1, error="HTTP 500: simulated")
        raise RuntimeError("simulated total outage")


# ---------------------------------------------------------------------------
# Fixture session: redundancy + high-value identifiers + a superseded fact
# ---------------------------------------------------------------------------
FILLER = [
    "Sure, sounds good to me.",
    "Okay let me take a look at that now.",
    "Nice, that build finished fine again.",
    "Thanks, I will check the logs in a moment.",
    "Alright, moving on to the next step here.",
]

HIGH_VALUE_LINES = [
    "The production API key lives at /srv/app/secrets/prod.env and must never be committed.",
    "We pinned pydantic==2.14.3 because 2.15 broke the serializer.",
    "The internal registry host is registry.internal:8443 for all package installs.",
    "Remember that the release deadline is 2026-12-04, no exceptions.",
    "The method we are reproducing is arXiv:2410.01234 from the OpenMemX paper.",
    "Preference: always use pnpm instead of npm in this repository.",
]

# Superseded fact: v2.1.0 -> v2.2.0 (later value must win).
SUPERSEDED_OLD = "We are deploying the v2.1.0 build to staging today."
SUPERSEDED_NEW = "Correction: the staging build is now v2.2.0, not the older one."


def build_fixture_session(session_id: str = "s001", n_filler_turns: int = 30) -> Session:
    """A deterministic session mixing redundancy, facts and an update.

    Sized so that the 0.25 token budget comfortably holds the six high-value
    sentences (~130 tokens) but not the redundant chatter, which is exactly the
    discrimination the extractive arm is supposed to make.
    """
    turns: list[Turn] = []
    idx = 0

    turns.append(Turn(idx=idx, role="user", text="Let's continue the deploy work for the billing service."))
    idx += 1
    for i, line in enumerate(HIGH_VALUE_LINES):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(idx=idx, role=role, text=line))
        idx += 1
        # Two filler turns between each fact: redundancy should dominate.
        for j in range(2):
            turns.append(
                Turn(idx=idx, role="user" if (i + j) % 2 == 0 else "assistant",
                     text=FILLER[(i * 2 + j) % len(FILLER)])
            )
            idx += 1

    turns.append(Turn(idx=idx, role="assistant", text=SUPERSEDED_OLD))
    idx += 1
    for k in range(max(0, n_filler_turns - 12)):
        turns.append(
            Turn(idx=idx, role="user" if k % 2 else "assistant",
                 text=FILLER[k % len(FILLER)] + " " + FILLER[(k + 2) % len(FILLER)])
        )
        idx += 1
    turns.append(Turn(idx=idx, role="user", text=SUPERSEDED_NEW))
    idx += 1

    facts = [
        Fact(fact_id="f01", kind="api_key_location", value="/srv/app/secrets/prod.env",
             domain="coding", first_turn=1, mentions=[1], salience="foreground",
             embedding="standalone", stability="stable"),
        Fact(fact_id="f02", kind="version", value="pydantic==2.14.3",
             domain="coding", first_turn=3, mentions=[3], salience="background",
             embedding="embedded", stability="stable"),
        Fact(fact_id="f03", kind="host", value="registry.internal:8443",
             domain="coding", first_turn=5, mentions=[5], salience="background",
             embedding="embedded", stability="stable"),
        Fact(fact_id="f04", kind="date", value="2026-12-04",
             domain="coding", first_turn=7, mentions=[7], salience="foreground",
             embedding="standalone", stability="stable"),
        Fact(fact_id="f05", kind="citation", value="arXiv:2410.01234",
             domain="coding", first_turn=9, mentions=[9], salience="background",
             embedding="deep_buried", stability="stable"),
        Fact(fact_id="f06", kind="preference", value="pnpm",
             domain="coding", first_turn=11, mentions=[11], salience="foreground",
             embedding="standalone", stability="stable"),
        Fact(fact_id="f07", kind="version", value="v2.2.0",
             domain="coding", first_turn=len(turns) - 3, mentions=[len(turns) - 3, len(turns) - 1],
             salience="background", embedding="embedded", stability="superseded",
             superseded_by="v2.2.0"),
    ]

    session = Session(
        session_id=session_id, domain="coding", turns=turns, facts=facts,
        seed=7, title="billing deploy session",
    )
    session.full_tokens = count_tokens(render_transcript(session))
    return session


@pytest.fixture()
def session() -> Session:
    return build_fixture_session()


def build_long_session(session_id: str = "s002", n_turns: int = 60) -> Session:
    """A larger session that forces multiple summary chunks (multi-level reduce)."""
    turns: list[Turn] = []
    for i in range(n_turns):
        if i % 6 == 3:
            text = (
                f"Turn {i}: the deploy config for service-{i} lives at "
                f"/srv/app/services/svc{i}/config/prod.yaml and the image tag is "
                f"svc{i}==1.{i}.0. The registry host is registry.internal:8443."
            )
        elif i % 11 == 5:
            text = (
                f"Turn {i}: important — the nightly job runs at 02:{i % 60:02d} on "
                f"worker-{i}.internal and writes to s3://bucket-{i}/daily/. "
                f"Remember to keep the retention window at 14 days."
            )
        else:
            text = (
                f"Turn {i}: status update. " + " ".join(FILLER[(i + k) % len(FILLER)] for k in range(3))
            )
        turns.append(Turn(idx=i, role="user" if i % 2 == 0 else "assistant", text=text))
    s = Session(session_id=session_id, domain="coding", turns=turns, facts=[], seed=11)
    s.full_tokens = count_tokens(render_transcript(s))
    return s


# ---------------------------------------------------------------------------
# base.py: budget fitting is exact
# ---------------------------------------------------------------------------
def test_truncate_respects_budget_exactly():
    text = render_transcript(build_fixture_session())
    for budget in (0, 1, 5, 37, 100, 250, 1000):
        out, truncated = truncate_to_token_budget(text, budget, marker=" …")
        assert count_tokens(out) <= budget, f"budget {budget} violated: {count_tokens(out)}"
        assert truncated is (budget < count_tokens(text))


def test_truncate_keeps_short_text_untouched():
    out, truncated = truncate_to_token_budget("hello there", 50)
    assert out == "hello there" and truncated is False


def test_split_by_token_budget_reassembles_content():
    text = render_transcript(build_fixture_session())
    pieces = split_by_token_budget(text, 60)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 60 for p in pieces)
    # No content is invented; only leading newlines are trimmed at boundaries.
    joined = "".join(pieces)
    assert count_tokens(joined) <= count_tokens(text)
    assert len(joined) >= len(text) - 3 * len(pieces)


def test_ledger_delta_counts_appended_records():
    client = FakeClient()
    assert ledger_delta(client, lambda: None)[1].cost_tokens == 0
    result, delta = ledger_delta(
        client, lambda: client.chat("summarizer", [{"role": "user", "content": "hi"}], max_tokens=10)
    )
    assert result
    assert delta.calls == 1
    assert delta.prompt_tokens > 0
    assert delta.completion_tokens == count_tokens(result)
    assert delta.cost_tokens == delta.prompt_tokens + delta.completion_tokens


def test_ledger_delta_tolerates_cache_hits():
    """A cache hit records zero tokens; the delta must be 0 and must not crash."""
    client = FakeClient()

    class CacheHitClient(FakeClient):
        def chat(self, role, messages, **kw):
            self.ledger.add(role=role, tag=kw.get("tag", ""), model="fake", ok=True,
                            cached=True, prompt_tokens=0, completion_tokens=0,
                            total_tokens=0, latency=0.0)
            return "- cached note"

    hit = CacheHitClient()
    hit.ledger = client.ledger

    def warm_call():
        return hit.chat("summarizer", [{"role": "user", "content": "warm"}], max_tokens=10)

    result, delta = ledger_delta(client, warm_call)
    assert result == "- cached note"
    assert delta.calls == 1
    assert delta.cache_hits == 1
    assert delta.cost_tokens == 0, "cache hits spend no tokens and must not crash accounting"


# ---------------------------------------------------------------------------
# Chunking keeps turn structure
# ---------------------------------------------------------------------------
def test_chunking_keeps_turn_boundaries_and_covers_session(session: Session):
    chunks = chunk_transcript(session, budget=400)
    assert len(chunks) > 1
    assert all(c.tokens <= 400 for c in chunks if c.tokens), "chunk exceeded budget"
    # Every turn's text appears somewhere in the concatenated chunks.
    combined = "".join(c.text for c in chunks)
    for turn in session.turns:
        assert turn.text[:40] in combined
    # Turn spans are monotone and cover the session.
    assert chunks[0].first_turn == 0
    assert chunks[-1].last_turn == session.turns[-1].idx


# ---------------------------------------------------------------------------
# Result field consistency (both arms)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["summary_hierarchical", "extractive"])
def test_result_fields_are_consistent(strategy, session):
    client = FakeClient(responder=FakeClient().chunk_aware_responder)
    comp = get_compressor(strategy)
    result = comp.compress(session, client)

    assert result.session_id == session.session_id
    assert result.strategy == strategy == comp.name
    assert result.compressed_tokens == count_tokens(result.text)
    assert result.original_tokens == session.full_tokens > 0
    expected_ratio = result.compressed_tokens / result.original_tokens
    assert result.ratio == pytest.approx(expected_ratio, rel=1e-9)
    assert 0.0 <= result.ratio <= 1.0
    assert result.cost_tokens >= 0


# ---------------------------------------------------------------------------
# Budget adherence (both arms)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["summary_hierarchical", "extractive"])
def test_budget_adherence(strategy, session):
    """The achieved ratio must respect the target, not just the prompt's request."""
    client = FakeClient(responder=FakeClient().chunk_aware_responder)
    comp = get_compressor(strategy)
    result = comp.compress(session, client)
    target = result.meta["target_ratio"]

    assert target == pytest.approx(0.25)
    assert result.compressed_tokens <= result.meta["budget_tokens"]
    assert result.ratio <= target + 0.01, (
        f"{strategy} ratio {result.ratio:.3f} exceeded target {target}"
    )


def test_summary_truncation_is_recorded_when_model_overshoots(session):
    """A summarizer that ignores the budget must be fitted and flagged."""

    def chatty(role, messages, kw):
        body = messages[-1]["content"]
        # Echo a large slice of the input so the note is far over budget.
        return " ".join(body.split()[:1200])

    client = FakeClient(responder=chatty)
    comp = HierarchicalSummaryCompressor()
    result = comp.compress(session, client)

    assert result.compressed_tokens <= result.meta["budget_tokens"]
    assert result.meta["truncated"] is True
    assert result.ratio <= result.meta["target_ratio"] + 0.01


# ---------------------------------------------------------------------------
# Hierarchical summary specifics
# ---------------------------------------------------------------------------
def test_summary_meta_and_prompt(session):
    client = FakeClient(responder=FakeClient().chunk_aware_responder)
    result = get_compressor("summary_hierarchical").compress(session, client)

    meta = result.meta
    assert meta["levels"] >= 2, "map + reduce at minimum"
    assert meta["chunks"] == len(chunk_transcript(session))
    assert meta["prompt_version"]
    assert meta["model"] == "fake/summarizer-1"
    assert meta["fallbacks"] == []
    # Every LLM call is accounted for: one per chunk, the reduce passes, and at
    # most one in-paradigm "expand" call when the note came back too thin.
    expected = meta["chunks"] + (meta["levels"] - 1) + (1 if meta.get("expanded") else 0)
    assert meta["llm_calls"] == expected == len(client.calls)
    assert meta["cost_tokens"] > 0

    # The map prompt must demand the exact fact kinds the probes test.
    map_prompt = client.calls[0]["messages"][0]["content"].lower()
    for needle in ("path", "package", "version pin", "host", "deadline", "preference",
                   "arxiv", "api key"):
        assert needle in map_prompt, f"map prompt does not mention {needle!r}"
    # max_tokens must be generous: reasoning models burn hidden tokens.
    assert all(c["max_tokens"] >= 900 for c in client.calls)


def test_summary_expands_when_the_first_pass_is_too_thin(session):
    """A terse summarizer is asked again in-paradigm before raw text is borrowed."""
    calls = {"n": 0}

    def terse_then_full(role, messages, kw):
        calls["n"] += 1
        if "CURRENT NOTE" in messages[-1]["content"]:
            return "\n".join(
                f"- USER: detail {i}: /srv/app/services/svc{i}/config.yaml pinned at "
                f"pkg{i}==2.{i}.0 on registry.internal:8443"
                for i in range(6)
            )
        return "- USER: the deploy is fine"

    result = HierarchicalSummaryCompressor().compress(session, FakeClient(responder=terse_then_full))
    assert result.meta["expanded"] is True
    assert result.meta["backfilled"] == 0, "in-paradigm expansion should be enough"
    assert result.compressed_tokens <= result.meta["budget_tokens"]
    assert result.meta["budget_utilization"] >= 0.75
    assert calls["n"] >= 3, "map + reduce + expand"


def test_summary_backfills_when_expansion_still_leaves_room(session):
    """If the model stays terse, leftover budget is spent verbatim and recorded."""
    result = HierarchicalSummaryCompressor().compress(
        session, FakeClient(responder=lambda role, messages, kw: "- USER: fine")
    )
    assert result.meta["backfilled"] > 0
    assert result.meta["budget_utilization"] >= 0.75
    assert result.compressed_tokens <= result.meta["budget_tokens"]
    # Backfilled lines are verbatim transcript with speaker labels, not invention.
    assert re.search(r"^(USER|ASSISTANT):", result.text, re.MULTILINE)


def test_summary_cost_matches_ledger_delta(session):
    client = FakeClient(responder=FakeClient().chunk_aware_responder)
    before = client.ledger.totals()["grand_total"]["total_tokens"]
    result = get_compressor("summary_hierarchical").compress(session, client)
    after = client.ledger.totals()["grand_total"]["total_tokens"]
    assert result.cost_tokens == after - before
    assert result.cost_tokens > 0


def test_summary_survives_a_failing_chunk(session):
    client = FlakyClient(n_failures=1, responder=FakeClient().chunk_aware_responder)
    result = get_compressor("summary_hierarchical").compress(session, client)

    assert result.meta["fallbacks"], "the failed chunk must be recorded"
    assert result.meta["fallbacks"][0]["stage"] == "map"
    assert result.text.strip(), "a chunk failure must not empty the memory"
    assert result.compressed_tokens == count_tokens(result.text)
    assert result.compressed_tokens <= result.meta["budget_tokens"]


def test_summary_survives_total_outage(session):
    result = get_compressor("summary_hierarchical").compress(session, ExplodingClient())
    assert result.text.strip(), "total outage must still yield a budgeted raw fallback"
    assert len(result.meta["fallbacks"]) >= 1
    assert result.compressed_tokens <= result.meta["budget_tokens"]
    assert result.cost_tokens == 0
    assert "circuit_breaker_open" in result.meta


def test_circuit_breaker_stops_calling_a_dead_gateway(session):
    """A quota-exhausted gateway must not be retried for every chunk and reduce.

    The free gateway returns HTTP 429 for every call once its allowance is spent.
    Without a breaker, each session burns minutes in exponential backoff. The
    breaker opens after two consecutive failures and the rest of the session
    falls back immediately — the result is still valid and the outage is visible.
    """
    long_session = build_long_session(n_turns=90)
    client = ExplodingClient()
    result = HierarchicalSummaryCompressor().compress(long_session, client)

    assert result.meta["chunks"] >= 3, "the fixture must have several chunks to skip"
    # At most 2 attempts (the breaker's threshold), not chunks + reduces.
    assert len(client.ledger.records) <= 2, (
        f"breaker did not stop the retries: {len(client.ledger.records)} calls"
    )
    assert result.meta["circuit_breaker_open"]
    assert any(f.get("error") == "circuit breaker open" for f in result.meta["fallbacks"])
    assert result.text.strip()
    assert result.compressed_tokens <= result.meta["budget_tokens"]


def test_circuit_breaker_closes_again_after_a_success(session):
    """One transient failure must not disable the whole session."""
    client = FlakyClient(n_failures=1, responder=FakeClient().chunk_aware_responder)
    result = HierarchicalSummaryCompressor().compress(session, client)
    assert "circuit_breaker_open" not in result.meta
    assert result.meta["llm_calls"] >= 2, "compression continued after the transient failure"


def test_summary_multi_chunk_tree():
    """A session bigger than the chunk window exercises map + grouped reduce."""
    long_session = build_long_session()
    client = FakeClient(responder=FakeClient().chunk_aware_responder)
    result = HierarchicalSummaryCompressor().compress(long_session, client)

    meta = result.meta
    assert meta["chunks"] >= 2, "the long session must be split into several windows"
    assert meta["levels"] >= 2
    assert meta["llm_calls"] == len(client.calls)
    assert meta["llm_calls"] >= meta["chunks"] + 1
    assert result.compressed_tokens <= meta["budget_tokens"]
    assert result.ratio <= meta["target_ratio"] + 0.01
    assert result.compressed_tokens == count_tokens(result.text)


def test_summary_keeps_bullets_whole_when_cutting(session):
    """Budget cuts must drop whole bullets, never leave a dangling half-fact."""

    def overlong(role, messages, kw):
        return "\n".join(
            f"- USER: fact {i} is /srv/app/data/file_{i}.yaml with version pkg{i}==1.{i}.0"
            for i in range(40)
        )

    result = HierarchicalSummaryCompressor().compress(session, FakeClient(responder=overlong))
    assert result.meta["truncated"] is True
    assert result.meta["bullets_kept"] < result.meta["bullets_total"]
    # Every emitted line must be a complete bullet from the model's output.
    for line in result.text.splitlines():
        stripped = line.rstrip(" …")
        if stripped:
            assert re.match(r"^- USER: fact \d+ is /srv/app/data/file_\d+\.yaml with version pkg\d+==1\.\d+\.0$",
                            stripped), f"partial bullet emitted: {line!r}"
    assert result.compressed_tokens <= result.meta["budget_tokens"]


def test_summary_bullet_priority_prefers_concrete_values():
    """The budget cut must not systematically delete the tail of the session."""
    comp = HierarchicalSummaryCompressor()
    units = [
        "- USER: sounds good, let's continue with the next step here",
        "- USER: okay thanks, that build finished fine again",
        "- ASSISTANT: Correction: the staging build is now v2.2.0 at /srv/app/release.yaml",
    ]
    priority = comp._bullet_priority(units)
    assert priority[0] == 2, "the concrete-value bullet must be admitted first"


def test_summary_bullet_priority_breaks_ties_towards_the_latest():
    """When equally informative bullets compete, the more recent one wins."""
    comp = HierarchicalSummaryCompressor()
    units = ["- USER: the host is registry.internal:8443", "- USER: the host is registry.internal:8443"]
    assert comp._bullet_priority(units)[0] == 1


def test_summary_budget_cut_does_not_always_drop_the_tail():
    """A positional cut would erase the updated value; importance-based must not."""
    units = [
        "- USER: we should probably take a look at the logs in a moment",
        "- USER: alright, moving on to the next step here",
        "- ASSISTANT: Correction: the staging build is now v2.2.0, not the older one.",
    ]
    from memfaith.compression.base import fit_units_to_budget

    text, truncated, kept = fit_units_to_budget(
        units, 25, priority=HierarchicalSummaryCompressor._bullet_priority(units)
    )
    assert truncated is True
    assert kept == 1
    assert "v2.2.0" in text, f"importance-based cut kept the wrong bullet: {text!r}"


# ---------------------------------------------------------------------------
# Extractive specifics
# ---------------------------------------------------------------------------
def test_extractive_is_deterministic(session):
    comp = ExtractiveCompressor()
    a = comp.compress(session)
    b = ExtractiveCompressor().compress(session)
    assert a.text == b.text
    assert a.compressed_tokens == b.compressed_tokens
    assert a.meta["kept_sentences"] == b.meta["kept_sentences"]
    assert a.meta["method"] == b.meta["method"]


def test_extractive_keeps_high_value_identifiers(session):
    result = ExtractiveCompressor().compress(session)
    text = result.text
    assert "/srv/app/secrets/prod.env" in text, "the API-key path must survive"
    assert "pydantic==2.14.3" in text, "the version pin must survive"
    assert "registry.internal:8443" in text, "the host:port must survive"
    assert "arXiv:2410.01234" in text, "the paper id must survive"
    assert "2026-12-04" in text, "the deadline must survive"


def test_extractive_drops_redundant_filler(session):
    result = ExtractiveCompressor().compress(session)
    # The filler lines repeat verbatim; at most a couple may survive, and the
    # output must be far smaller than the input.
    survivors = sum(1 for f in FILLER if f in result.text)
    assert survivors <= 2, f"{survivors} filler lines survived dedup"
    assert result.ratio <= 0.25 + 0.01


def test_extractive_prefers_latest_value_of_superseded_fact():
    """A later value must beat the earlier one when both compete for budget."""
    turns = [
        Turn(idx=0, role="assistant", text="We are deploying the v2.1.0 build to staging today."),
        Turn(idx=1, role="user", text="ok"),
        Turn(idx=2, role="user", text="Correction: the staging build is now v2.2.0, not the older one."),
    ]
    s = Session(session_id="s-super", domain="coding", turns=turns, facts=[], seed=1)
    s.full_tokens = count_tokens(render_transcript(s))

    # A 0.6 ratio leaves room for roughly one of the two competing sentences.
    result = ExtractiveCompressor(target_ratio=0.6).compress(s)
    assert "v2.2.0" in result.text, f"latest value missing from: {result.text!r}"
    assert "v2.1.0" not in result.text, f"stale value survived: {result.text!r}"


def test_extractive_meta_fields(session):
    result = ExtractiveCompressor().compress(session)
    meta = result.meta
    assert meta["method"]
    assert meta["llmlingua"] is False
    assert meta["total_sentences"] >= meta["kept_sentences"] > 0
    assert meta["budget_tokens"] == int(session.full_tokens * 0.25)
    assert meta["cost_tokens"] == 0, "the extractive arm must spend zero LLM tokens"
    assert result.cost_tokens == 0


def test_extractive_keeps_speaker_attribution(session):
    text = ExtractiveCompressor().compress(session).text
    assert re.search(r"^(USER|ASSISTANT):", text, re.MULTILINE)


def test_sentence_splitter_does_not_shred_identifiers():
    line = "We pinned pydantic==2.14.3 and the docs at https://example.com/a.b work, e.g. in /srv/app/x.py."
    parts = split_sentences(line)
    assert any("pydantic==2.14.3" in p for p in parts)
    assert any("https://example.com/a.b" in p for p in parts)
    assert any("/srv/app/x.py" in p for p in parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_exposes_both_arms():
    assert set(STRATEGIES) == {"summary_hierarchical", "extractive"}
    for name, cls in STRATEGIES.items():
        comp = get_compressor(name)
        assert isinstance(comp, cls)
        assert comp.name == name
        assert callable(comp.compress)


def test_get_compressor_accepts_overrides():
    comp = get_compressor("extractive", target_ratio=0.1)
    assert comp.target_ratio == pytest.approx(0.1)


def test_get_compressor_rejects_unknown_name():
    with pytest.raises(KeyError) as exc:
        get_compressor("magic_compression")
    assert "magic_compression" in str(exc.value)
    assert "extractive" in str(exc.value)


# ---------------------------------------------------------------------------
# Measured ratios on the fixture (reported to the Lead)
# ---------------------------------------------------------------------------
def test_fixture_ratios_are_measurable(session):
    """Both arms land near the 0.25 target on the fixture; print for the report."""
    summary = get_compressor("summary_hierarchical").compress(
        session, FakeClient(responder=FakeClient().chunk_aware_responder)
    )
    extractive = get_compressor("extractive").compress(session)

    assert session.full_tokens > 300
    assert 0.05 < extractive.ratio <= 0.26
    assert 0.02 < summary.ratio <= 0.26
    assert extractive.compressed_tokens < session.full_tokens
    assert summary.compressed_tokens < session.full_tokens
