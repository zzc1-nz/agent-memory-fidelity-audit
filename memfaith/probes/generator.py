"""Probe questions that make compression fidelity measurable.

Three families, chosen because they fail differently under compression:

``high_level``
    Gist questions ("what was the goal, which stack won"). A competent summary
    should keep these. If an arm fails here, the compressor is simply bad — not
    "lossy in an interesting way".

``specific_fact``
    One exact literal (a path, a port, a date, a citation id). This is where
    abstraction loses information: a summarizer that writes "the API key is kept
    in a secrets file" has destroyed the answer while reading as perfectly
    coherent. Each probe depends on exactly one fact.

``multi_hop``
    Requires combining >= 2 facts, at least one of which is likely compressed
    away. This is the family that catches "each fact survived but their
    relationship did not".

Superseded facts get special handling: the probe asks for the CURRENT value and
records the stale value as a trap. An arm that keeps the old mention and drops
the update fails here — the most dangerous failure mode, because the answer is
confidently wrong rather than obviously missing.
"""

from __future__ import annotations

import json
import os
import random

from memfaith.schema import Fact, Probe, Session

# Question phrasing per fact kind. Kept short and literal so the reference answer
# can be matched by a judge without paraphrase tolerance games.
KIND_QUESTIONS = {
    "package": "Which package manager did the team standardise on?",
    "path": "What is the full path of the production environment file?",
    "api_key_location": "Where is the API key loaded from?",
    "host": "What is the staging database host and port?",
    "version": "Which version was pinned for this work?",
    "constraint": "What hard constraint applies to any solution we ship?",
    "citation": "Which paper must the related-work section cite?",
    "dataset_source": "What is the canonical source of the evaluation dataset?",
    "date": "What is the relevant deadline or renewal date?",
    "preference": "What standing preference did the user state?",
}

# Multi-hop pairings: (fact kind A, fact kind B, question template).
# The template takes (a_value, b_value) but asks the model to supply both.
MULTIHOP_TEMPLATES = [
    ("package", "host",
     "We settled on a package manager and a staging database. Which package manager did we "
     "standardise on, and which host:port serves staging?"),
    ("path", "api_key_location",
     "There are two credential locations in play. Which path holds the production environment "
     "file, and where is the API key actually loaded from?"),
    ("citation", "date",
     "For the write-up: which paper must we cite, and what is the submission deadline?"),
    ("dataset_source", "version",
     "Which dataset source are we evaluating on, and which version of the harness are we using?"),
    ("preference", "constraint",
     "Which standing preference did the user give, and what hard limit applies?"),
    ("version", "host",
     "Which version did we pin, and what is the staging database host:port?"),
    ("package", "version",
     "Which package manager did we standardise on, and which version did we pin?"),
    ("path", "host",
     "Which path holds the production environment file, and what is the staging database "
     "host:port?"),
]


def _fact_question(fact: Fact) -> str:
    base = KIND_QUESTIONS.get(fact.kind, "What was the exact value recorded for this detail?")
    if fact.stability == "superseded":
        return base + " (Answer with the value currently in effect, not any earlier one.)"
    return base


def _reference_for(fact: Fact) -> str:
    """The literal the judge compares against.

    For superseded facts this is the CURRENT value; the stale value lives in
    ``Probe.notes`` so the failure analysis can detect a stale-value answer.
    """
    return fact.value


def _build_high_level(session: Session, idx: int, rng: random.Random,
                      variant: int = 0) -> Probe:
    """Gist probe: goal + settled choices. Answerable from a good summary.

    Three variants so a session's high-level probes are not literally identical:
    one about the objective, one about the settled technical choices, one about
    what was left unresolved. They draw on different facts so they are genuinely
    distinct questions rather than the same question repeated.
    """
    goal = session.title.split(": ", 1)[-1]
    stable = [f for f in session.facts if f.stability == "stable"]
    superseded = [f for f in session.facts if f.stability == "superseded"]

    if variant % 3 == 0:
        facts = stable[:3] or session.facts[:3]
        return Probe(
            probe_id=f"{session.session_id}-p{idx:02d}",
            session_id=session.session_id,
            kind="high_level",
            question=("Summarise this session: what was the overall objective, and which concrete "
                      "choices were settled on along the way?"),
            reference_answer=f"Objective: {goal}. Settled on: " + "; ".join(f.value for f in facts),
            fact_ids=[f.fact_id for f in facts],
            domain=session.domain, hops=1,
            notes="Gist probe (objective + choices); a competent summary should retain this.",
        )

    if variant % 3 == 1:
        facts = stable[3:6] or stable[:2] or session.facts[:2]
        return Probe(
            probe_id=f"{session.session_id}-p{idx:02d}",
            session_id=session.session_id,
            kind="high_level",
            question=("What were the main technical or operational decisions recorded in this "
                      "session, and what were they for?"),
            reference_answer="; ".join(f.value for f in facts),
            fact_ids=[f.fact_id for f in facts],
            domain=session.domain, hops=1,
            notes="Gist probe (decisions); summary should keep the decision set.",
        )

    # variant 2: what changed / what is now current — tests abstraction of updates.
    # The reference is the literal current value (not a sentence about it) so the
    # judge compares literals, and the probe stays answerable from the transcript.
    target = superseded[0] if superseded else (stable[0] if stable else session.facts[0])
    return Probe(
        probe_id=f"{session.session_id}-p{idx:02d}",
        session_id=session.session_id,
        kind="high_level",
        question=("Did anything in this session get updated or revised? Describe the change and "
                  "what the current state is."),
        reference_answer=target.value,
        fact_ids=[target.fact_id],
        domain=session.domain, hops=1,
        notes=("Gist probe (updates); tests whether abstraction preserves that a change happened. "
               f"Current value is {target.value!r}"
               + (f"; stale value {target.superseded_old_value!r} must not be reported."
                  if target.stability == "superseded" else ".")),
    )


def _build_specific(session: Session, idx: int, rng: random.Random, fact: Fact) -> Probe:
    notes = ""
    if fact.stability == "superseded":
        notes = (f"SUPERSEDED TRAP: current value is {fact.value!r}; the stale value "
                 f"{fact.superseded_old_value!r} appears earlier and must NOT be accepted.")
    return Probe(
        probe_id=f"{session.session_id}-p{idx:02d}",
        session_id=session.session_id,
        kind="specific_fact",
        question=_fact_question(fact),
        reference_answer=_reference_for(fact),
        fact_ids=[fact.fact_id],
        domain=session.domain,
        hops=1,
        notes=notes,
    )


def _build_multihop(session: Session, idx: int, rng: random.Random,
                    pair: tuple[str, str]) -> Probe | None:
    """Combine two facts of the requested kinds, if the session has both."""
    kind_a, kind_b = pair
    by_kind: dict[str, list[Fact]] = {}
    for f in session.facts:
        by_kind.setdefault(f.kind, []).append(f)
    cands_a = by_kind.get(kind_a, [])
    cands_b = by_kind.get(kind_b, [])
    if not cands_a or not cands_b:
        return None
    a = rng.choice(cands_a)
    b = rng.choice(cands_b)
    if a.fact_id == b.fact_id:
        return None

    template = next(t for (ka, kb, t) in MULTIHOP_TEMPLATES if ka == kind_a and kb == kind_b)
    notes = "Multi-hop: requires both facts."
    if a.stability == "superseded" or b.stability == "superseded":
        stale = a.superseded_old_value if a.stability == "superseded" else b.superseded_old_value
        notes += f" Contains a superseded fact (stale value {stale!r} must not be used)."
    return Probe(
        probe_id=f"{session.session_id}-p{idx:02d}",
        session_id=session.session_id,
        kind="multi_hop",
        question=template,
        reference_answer=f"{a.value}; {b.value}",
        fact_ids=[a.fact_id, b.fact_id],
        domain=session.domain,
        hops=2,
        notes=notes,
    )


def build_probes(sessions: list[Session], per_session: int = 10,
                 mix: tuple[int, int, int] = (3, 5, 2)) -> list[Probe]:
    """Build ``per_session`` probes for every session, deterministically.

    ``mix`` is (high_level, specific_fact, multi_hop) and must sum to
    ``per_session``. Specific-fact probes are drawn to cover the designed
    covariates (embedding depth, salience, stability) rather than sampling at
    random, so the failure analysis has enough cells to say something.
    """
    n_high, n_spec, n_multi = mix
    if n_high + n_spec + n_multi != per_session:
        raise ValueError(f"mix {mix} does not sum to per_session={per_session}")

    probes: list[Probe] = []
    for s_index, session in enumerate(sessions, 1):
        rng = random.Random(20261110 * 1000 + s_index)
        idx = 0

        # --- high-level ----------------------------------------------------
        for v in range(n_high):
            probes.append(_build_high_level(session, idx, rng, variant=v))
            idx += 1

        # --- specific facts -------------------------------------------------
        # Prefer coverage: sort facts so the superseded one is always included,
        # then alternate across embedding depths and salience.
        facts = list(session.facts)
        superseded = [f for f in facts if f.stability == "superseded"]
        rest = [f for f in facts if f.stability != "superseded"]
        rng.shuffle(rest)
        # Interleave deep_buried / embedded / standalone so the sample spans depths.
        buckets: dict[str, list[Fact]] = {}
        for f in rest:
            buckets.setdefault(f.embedding, []).append(f)
        ordered: list[Fact] = list(superseded)
        while len(ordered) < len(facts):
            for depth in ("deep_buried", "embedded", "standalone"):
                if buckets.get(depth):
                    ordered.append(buckets[depth].pop(0))
        chosen = ordered[:n_spec]
        for fact in chosen:
            probes.append(_build_specific(session, idx, rng, fact))
            idx += 1

        # --- multi-hop -------------------------------------------------------
        available_pairs = []
        kinds = {f.kind for f in session.facts}
        for pair in MULTIHOP_TEMPLATES:
            if pair[0] in kinds and pair[1] in kinds:
                available_pairs.append((pair[0], pair[1]))
        rng.shuffle(available_pairs)
        built = 0
        for pair in available_pairs:
            if built >= n_multi:
                break
            p = _build_multihop(session, idx, rng, pair)
            if p is not None:
                probes.append(p)
                idx += 1
                built += 1
        # If the session's fact mix cannot supply enough multi-hop pairs, fall
        # back to combining any two distinct facts so the count contract holds.
        while built < n_multi:
            f1, f2 = rng.sample(session.facts, 2)
            probes.append(Probe(
                probe_id=f"{session.session_id}-p{idx:02d}",
                session_id=session.session_id,
                kind="multi_hop",
                question=(f"Two details from this session: {KIND_QUESTIONS.get(f1.kind, 'value one')} "
                          f"And: {KIND_QUESTIONS.get(f2.kind, 'value two')}"),
                reference_answer=f"{f1.value}; {f2.value}",
                fact_ids=[f1.fact_id, f2.fact_id],
                domain=session.domain,
                hops=2,
                notes="Multi-hop fallback pairing.",
            ))
            idx += 1
            built += 1

    return probes


def save_probes(probes: list[Probe], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in probes:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_probes(path: str) -> list[Probe]:
    out: list[Probe] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Probe(**json.loads(line)))
    return out
