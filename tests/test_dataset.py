"""Dataset invariants. These are the properties the audit's validity rests on."""

from __future__ import annotations

import json

import pytest

from memfaith.dataset.generator import (
    build_sessions,
    load_sessions,
    redundancy_stats,
    render_transcript,
    save_sessions,
)
from memfaith.tokens import count_tokens

SEED = 20261110


@pytest.fixture(scope="module")
def sessions():
    return build_sessions(30, SEED)


def test_session_count_and_ids(sessions):
    assert len(sessions) == 30
    assert [s.session_id for s in sessions] == [f"s{i:03d}" for i in range(1, 31)]


def test_turn_counts_within_range(sessions):
    for s in sessions:
        assert 20 <= len(s.turns) <= 30, f"{s.session_id} has {len(s.turns)} turns"
        assert [t.idx for t in s.turns] == list(range(len(s.turns)))


def test_domain_mix(sessions):
    counts: dict[str, int] = {}
    for s in sessions:
        counts[s.domain] = counts.get(s.domain, 0) + 1
    assert set(counts) == {"coding", "research", "daily"}
    # 40/30/30 target with rounding tolerance
    assert 10 <= counts["coding"] <= 14, counts
    assert 7 <= counts["research"] <= 11, counts
    assert 7 <= counts["daily"] <= 11, counts


def test_determinism_same_seed():
    a = build_sessions(5, SEED)
    b = build_sessions(5, SEED)
    ja = [json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) for s in a]
    jb = [json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) for s in b]
    assert ja == jb


def test_different_seeds_differ():
    a = build_sessions(5, SEED)
    b = build_sessions(5, SEED + 1)
    ja = [json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) for s in a]
    jb = [json.dumps(s.to_dict(), ensure_ascii=False, sort_keys=True) for s in b]
    assert ja != jb


def test_full_tokens_matches_render(sessions):
    for s in sessions:
        assert s.full_tokens == count_tokens(render_transcript(s))
        assert s.full_tokens > 300


def test_every_fact_value_appears_verbatim(sessions):
    """The failure analysis substring-checks fact values against compressed text.

    If a value were paraphrased in the transcript, "fact lost" would be
    unmeasurable, so this invariant is load-bearing.
    """
    for s in sessions:
        text = render_transcript(s)
        for f in s.facts:
            assert f.value in text, f"{s.session_id}/{f.fact_id} value {f.value!r} not in transcript"


def test_fact_counts_and_mentions(sessions):
    for s in sessions:
        assert 6 <= len(s.facts) <= 10, f"{s.session_id} has {len(s.facts)} facts"
        for f in s.facts:
            assert f.mentions, f"{s.session_id}/{f.fact_id} has no mentions"
            assert 1 <= f.frequency <= 3
            assert f.first_turn == min(f.mentions)
            # mentions must point at turns that really contain the value
            for idx in f.mentions:
                assert f.value in s.turns[idx].text


def test_redundancy_share(sessions):
    ratios = [redundancy_stats(s)["redundancy_ratio"] for s in sessions]
    mean = sum(ratios) / len(ratios)
    assert mean >= 0.40, f"mean redundancy {mean:.3f} too low; the audit needs filler to remove"
    assert min(ratios) >= 0.25, f"a session has only {min(ratios):.3f} redundancy"


def test_every_session_has_one_superseded_fact(sessions):
    for s in sessions:
        sup = [f for f in s.facts if f.stability == "superseded"]
        assert len(sup) == 1, f"{s.session_id} has {len(sup)} superseded facts"
        f = sup[0]
        text = render_transcript(s)
        # The current value must be present, the stale value must also appear
        # earlier (that is the trap), and they must differ.
        assert f.value in text
        assert f.superseded_old_value
        assert f.superseded_old_value in text
        assert f.superseded_old_value != f.value
        # The update must come after the stale mention.
        stale_turn = next(i for i, t in enumerate(s.turns) if f.superseded_old_value in t.text)
        current_turn = min(f.mentions)
        assert current_turn > stale_turn, "the update must arrive after the stale value"


def test_covariate_variety(sessions):
    """The failure analysis needs cells: both salience values, all depths."""
    embeddings = {f.embedding for s in sessions for f in s.facts}
    saliences = {f.salience for s in sessions for f in s.facts}
    kinds = {f.kind for s in sessions for f in s.facts}
    assert {"standalone", "embedded", "deep_buried"} <= embeddings
    assert {"foreground", "background"} <= saliences
    assert len(kinds) >= 8


def test_save_load_roundtrip(tmp_path, sessions):
    path = tmp_path / "sessions.jsonl"
    save_sessions(sessions[:3], str(path))
    loaded = load_sessions(str(path))
    assert len(loaded) == 3
    for a, b in zip(sessions[:3], loaded):
        assert a.session_id == b.session_id
        assert a.full_tokens == b.full_tokens
        assert render_transcript(a) == render_transcript(b)
        assert len(a.facts) == len(b.facts)
        assert a.facts[0].value == b.facts[0].value


def test_deep_buried_facts_are_actually_buried(sessions):
    """A 'deep_buried' fact should sit in a turn much longer than a standalone one."""
    buried_lengths, standalone_lengths = [], []
    for s in sessions:
        for f in s.facts:
            turn = s.turns[min(f.mentions)]
            n = count_tokens(turn.text)
            if f.embedding == "deep_buried":
                buried_lengths.append(n)
            elif f.embedding == "standalone":
                standalone_lengths.append(n)
    if buried_lengths and standalone_lengths:
        avg_buried = sum(buried_lengths) / len(buried_lengths)
        avg_standalone = sum(standalone_lengths) / len(standalone_lengths)
        assert avg_buried > avg_standalone * 2, (
            f"deep_buried avg {avg_buried:.0f} vs standalone {avg_standalone:.0f}")
