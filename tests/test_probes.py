"""Probe invariants: every probe must be answerable, unambiguous, and correctly keyed."""

from __future__ import annotations

import json

import pytest

from memfaith.dataset.generator import build_sessions, render_transcript
from memfaith.probes.generator import build_probes, load_probes, save_probes

SEED = 20261110
MIX = (3, 5, 2)
PER_SESSION = sum(MIX)


@pytest.fixture(scope="module")
def data():
    sessions = build_sessions(10, SEED)
    probes = build_probes(sessions, PER_SESSION, MIX)
    return sessions, probes


def test_probe_count(data):
    sessions, probes = data
    assert len(probes) == len(sessions) * PER_SESSION


def test_probe_mix(data):
    _, probes = data
    counts: dict[str, int] = {}
    for p in probes:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    n = len(probes)
    assert counts["high_level"] == n * 3 // 10
    assert counts["specific_fact"] == n * 5 // 10
    assert counts["multi_hop"] == n * 2 // 10


def test_probe_ids_unique_and_well_formed(data):
    _, probes = data
    ids = [p.probe_id for p in probes]
    assert len(ids) == len(set(ids))
    for p in probes:
        assert p.probe_id.startswith(p.session_id + "-p")


def test_probe_fact_ids_exist(data):
    sessions, probes = data
    index = {s.session_id: {f.fact_id for f in s.facts} for s in sessions}
    for p in probes:
        assert p.fact_ids, f"{p.probe_id} has no fact_ids"
        assert set(p.fact_ids) <= index[p.session_id], f"{p.probe_id} references unknown facts"


def test_specific_fact_probes_have_one_fact(data):
    _, probes = data
    for p in probes:
        if p.kind == "specific_fact":
            assert len(p.fact_ids) == 1, p.probe_id
            assert p.hops == 1


def test_multi_hop_probes_need_multiple_facts(data):
    _, probes = data
    for p in probes:
        if p.kind == "multi_hop":
            assert len(p.fact_ids) >= 2, p.probe_id
            assert p.hops >= 2, p.probe_id


def test_reference_answers_are_nonempty_and_literal(data):
    sessions, probes = data
    index = {s.session_id: s for s in sessions}
    for p in probes:
        assert p.reference_answer.strip(), p.probe_id
        # For specific-fact probes the reference must be the fact's literal value,
        # which is what makes substring judging meaningful.
        if p.kind == "specific_fact":
            fact = index[p.session_id].fact_index()[p.fact_ids[0]]
            assert p.reference_answer == fact.value, p.probe_id


def test_superseded_probes_expect_current_value(data):
    """The stale value must never be the reference answer."""
    sessions, probes = data
    index = {s.session_id: s.fact_index() for s in sessions}
    checked = 0
    for p in probes:
        for fid in p.fact_ids:
            fact = index[p.session_id][fid]
            if fact.stability == "superseded":
                checked += 1
                assert fact.value in p.reference_answer, p.probe_id
                assert fact.superseded_old_value not in p.reference_answer, p.probe_id
    assert checked > 0, "no superseded facts were probed; the update axis is untested"


def test_high_level_questions_are_distinct(data):
    sessions, probes = data
    for s in sessions:
        hl = [p for p in probes if p.session_id == s.session_id and p.kind == "high_level"]
        assert len({p.question for p in hl}) == len(hl), f"{s.session_id} repeats a gist question"


def test_questions_are_answerable_from_transcript(data):
    """A probe whose reference is absent from the history is an unfair test."""
    sessions, probes = data
    index = {s.session_id: s for s in sessions}
    for p in probes:
        text = render_transcript(index[p.session_id])
        if p.kind == "specific_fact":
            assert p.reference_answer in text, p.probe_id
        elif p.kind == "high_level" and p.reference_answer.startswith("Objective:"):
            # Composite gist answer: "Objective: <goal>. Settled on: <v1>; <v2>; <v3>".
            # The goal clause is generated prose; every settled value must appear
            # literally in the history.
            settled = p.reference_answer.split("Settled on: ", 1)[-1]
            for part in settled.split("; "):
                part = part.strip()
                assert part in text, f"{p.probe_id}: {part!r} not in transcript"
        else:
            # Every literal listed in the reference must appear in the history.
            for part in p.reference_answer.split(";"):
                part = part.strip()
                if part:
                    assert part in text, f"{p.probe_id}: {part!r} not in transcript"


def test_determinism(data):
    sessions, probes = data
    again = build_probes(sessions, PER_SESSION, MIX)
    a = [json.dumps(p.to_dict(), sort_keys=True) for p in probes]
    b = [json.dumps(p.to_dict(), sort_keys=True) for p in again]
    assert a == b


def test_mix_validation():
    sessions = build_sessions(1, SEED)
    with pytest.raises(ValueError):
        build_probes(sessions, 10, (3, 5, 3))


def test_save_load_roundtrip(tmp_path, data):
    _, probes = data
    path = tmp_path / "probes.jsonl"
    save_probes(probes, str(path))
    loaded = load_probes(str(path))
    assert len(loaded) == len(probes)
    assert loaded[0].probe_id == probes[0].probe_id
    assert loaded[0].reference_answer == probes[0].reference_answer
