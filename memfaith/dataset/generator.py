"""Synthetic long agent-user interaction histories with controlled fidelity stakes.

Design goal
-----------
The benchmark needs histories in which the *interesting* thing can happen: a
compressor must be able to remove a lot of text, and the removed text must split
cleanly into two classes:

* **redundancy** — content that is said repeatedly and carries no unique fact.
  Removing it is a pure win. We target >=45% of raw tokens.
* **critical low-frequency facts** — mentioned once or twice, and load-bearing for
  a downstream task. Removing them is a silent failure.

Every session is therefore assembled from a domain-specific backbone (what the
agent and user are actually doing) with redundancy injected between the
informative turns. Critical facts are placed at controlled embedding depths so
the failure analysis has real covariates rather than a post-hoc story.

The update/forgetting axis
--------------------------
Each session contains at least one **superseded** fact: an early turn states the
old value, a late turn explicitly overwrites it. ``Fact.value`` is always the
*current* value (what probes must expect) and ``Fact.superseded_old_value`` keeps
the stale one. This distinguishes three distinct failure modes in the analysis:

  * fact lost entirely            -> probe unanswerable
  * fact kept but stale value won -> confidently wrong (worst)
  * fact kept correctly           -> probe passes

Determinism
-----------
``build_sessions(n, seed)`` is a pure function of ``(n, seed)``. Every random
choice goes through ``random.Random(seed * 1000 + index)``, so the same seed
reproduces byte-identical JSON. No wall clock, no LLM, no environment input.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from memfaith.schema import Fact, Session, Turn
from memfaith.tokens import count_tokens

# ---------------------------------------------------------------------------
# Vocabulary. Plain data, so sessions read like real transcripts.
# ---------------------------------------------------------------------------

# Redundant chatter: the class that *should* be compressed away. Deliberately
# information-free but domain-plausible, and verbose the way real agent logs are.
REDUNDANT_CHATTER: dict[str, list[str]] = {
    "coding": [
        "Quick status check: the build is still running and nothing has changed since the last update. "
        "Same three jobs in the queue, same order as the last two runs, no new warnings in the log.",
        "I re-ran the full test suite just to be safe and got exactly the same result as before. "
        "No new failures, no flaky tests this time, and the runtime is within a second of the previous run.",
        "Still waiting on CI. Nothing has moved in the pipeline, the cache is warm, and the artifacts "
        "from the earlier stage are still valid, so there is nothing for us to do but wait.",
        "Confirmed again that the working tree is clean and there are no uncommitted changes at this point. "
        "The last commit is the one we already reviewed, and the branch is in sync with the remote.",
        "Let me restate where we are so nothing gets lost: same branch, same task, same approach, "
        "no new blockers. We have not changed the dependency list since the last time we looked at it.",
        "Everything looks fine on my side. No errors in the logs since the previous check, the health "
        "endpoint is green, and the metrics are flat within normal variance for this time of day.",
        "Another pass over the diff shows the same files as before, nothing unexpected appeared, "
        "and the line counts are close enough to the previous review that I am not worried about it.",
        "I verified once more that the environment variables are still set the way we left them, "
        "the toolchain resolves to the same versions, and the shell profile has not been touched.",
        "The watcher is quiet, the terminal is idle, and nothing has moved since the last checkpoint. "
        "I will keep an eye on it but I do not expect any surprises in the next few minutes.",
        "Re-running the same command produced the same output, which is reassuring but not new information. "
        "The exit code is zero, the output is byte-identical, and the timing is unchanged.",
        "No action needed from you right now, this is just the routine status note for this stage. "
        "I will ping you the moment anything actually changes, so you do not need to keep checking.",
        "Same story as before: the pipeline is healthy, the queue is short, and we are on schedule. "
        "If that changes I will say so explicitly rather than making you read the logs yourself.",
    ],
    "research": [
        "Quick status check: no new papers have appeared in the saved search since the last refresh. "
        "The alert is still configured correctly, the query has not drifted, and the results are stable.",
        "I re-read the same section again and my understanding has not changed from the previous pass. "
        "The claims are the same, the caveats are the same, and the open questions are the same.",
        "The citation manager is still synced and shows the same library as before, nothing new. "
        "No duplicate keys, no broken links, and the export format is still the one the template expects.",
        "Just confirming the notes are unchanged: same sources, same summaries, same open questions. "
        "I have not added anything since the last review and I do not think anything needs removing.",
        "The search results are identical to the previous query, which is expected at this point. "
        "The ranking is stable, the top hits are the same, and the abstracts have not been revised.",
        "Let me restate the current state so we stay aligned: same topic, same scope, no new constraints. "
        "The plan is unchanged and the order of the remaining reading is exactly what we agreed on.",
        "No updates on my side. The reading list is stable and the backlog has not moved. "
        "I will keep working through it in the same order unless you tell me to reprioritise.",
        "Another check of the reference list shows the same entries as the last time we looked. "
        "The formatting is consistent and nothing has been retracted or updated in the meantime.",
        "I verified the citation keys once more and they are still consistent with the notes. "
        "Every claim in the draft maps to a source, and no source is cited twice under different keys.",
        "The library watcher is quiet and nothing has been added or removed since the checkpoint. "
        "If a new version of anything shows up I will flag it, but I do not expect that today.",
        "Re-running the same query returned the same set of results, as expected. "
        "The counts are identical, the ranking is identical, and the export is byte-for-byte the same.",
        "Same story as before: the reading is on track and there are no surprises in the sources. "
        "I will keep the notes updated as I go so we do not have to reconstruct this later.",
    ],
    "daily": [
        "Quick status check: nothing has changed on the calendar since we last looked at it. "
        "Same events, same times, no new invitations and nothing cancelled in the last few hours.",
        "I re-checked the same list and it still has the same items in the same order. "
        "Nothing has been completed, nothing has been added, and the priorities are unchanged.",
        "Just confirming the plan is unchanged: same tasks, same times, same priorities. "
        "If anything shifts I will tell you right away rather than letting it slide silently.",
        "No new messages or reminders have come in since the previous update. "
        "The inbox is at the same count as before and nothing in it looks time-sensitive.",
        "Let me restate where we are so nothing gets lost: same goals, same schedule, no new constraints. "
        "We are still working with the same set of assumptions we started the day with.",
        "Everything looks fine on my side, nothing urgent has appeared in the last few hours. "
        "The routine items are still routine and the important ones are still on track.",
        "I verified once more that the notes are still accurate and nothing needs updating. "
        "The details match what we discussed and I have not had to correct anything.",
        "Another pass over the todo list shows exactly the same items as before. "
        "Nothing has been quietly dropped and nothing has grown in scope since the last review.",
        "I want to re-confirm we are still on the same page about the priorities and the order. "
        "Nothing in the last few hours has given me a reason to suggest changing them.",
        "The reminder watcher is quiet and no new items have been added since the checkpoint. "
        "I will keep the schedule as it is unless something genuinely urgent comes up.",
        "Re-reading the same note produced the same understanding, no new details surfaced. "
        "The wording is still accurate and the open items are still the ones we listed.",
        "Same story as before: the schedule holds and nothing has slipped. "
        "I will keep checking in at the same cadence so you do not have to ask.",
    ],
}

# Filler used to pad assistant turns so a fact can be buried far from any cue.
ASSISTANT_PADDING = (
    "I went through the relevant logs and configuration once more to be sure, compared them "
    "against the previous checkpoint, and found the same picture as before with no new anomalies "
    "worth flagging. The surrounding context is unchanged, the environment is stable, and nothing "
    "in the recent output suggests we need to revisit any earlier decision. I have noted where we "
    "are so that if anything does change later we can tell exactly which step introduced it."
)

USER_SMALL_TALK = [
    "Thanks, that helps.",
    "Okay, sounds good to me.",
    "Got it, let's keep going.",
    "Makes sense, continue please.",
    "Appreciate the update.",
    "Fine by me, carry on.",
    "Understood, thanks for checking.",
    "Alright, keep me posted.",
]


# ---------------------------------------------------------------------------
# Fact blueprints: how each kind of fact is stated and asked about.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactBlueprint:
    kind: str
    values: tuple[str, ...]
    statement: str          # uses {value}
    question: str
    salience: str = "background"
    embedding: str = "embedded"


CODING_FACTS: list[FactBlueprint] = [
    FactBlueprint("package", ("uv", "poetry", "pip-tools", "pdm"),
                  "We standardised on {value} as the package manager for this repo.",
                  "Which package manager did the team standardise on?",
                  "foreground", "standalone"),
    FactBlueprint("path", ("/srv/app/.env.prod", "/opt/svc/config/.env.production", "/etc/app/env.prod"),
                  "The production environment file lives at {value}.",
                  "Where does the production environment file live?",
                  "foreground", "standalone"),
    FactBlueprint("api_key_location", ("~/.config/svc/token.env", "/run/secrets/svc_api_key", "~/secrets/openai.key"),
                  "The API key is loaded from {value} and never committed to the repo.",
                  "Where is the API key loaded from?",
                  "background", "deep_buried"),
    FactBlueprint("host", ("pg-staging-7.internal:6432", "db-stage-02.internal:5433", "staging-pg.internal:6432"),
                  "The staging database is reachable at {value}.",
                  "What is the staging database host and port?",
                  "background", "embedded"),
    FactBlueprint("version", ("v2.2.0", "v3.1.4", "v1.9.7"),
                  "We pinned the runtime to {value} for the migration.",
                  "Which runtime version did we pin for the migration?",
                  "background", "deep_buried"),
    FactBlueprint("constraint", ("no-new-deps", "python>=3.12", "no-native-extensions"),
                  "There is a hard constraint: {value} must hold for any solution we ship.",
                  "What hard constraint applies to the solution?",
                  "foreground", "embedded"),
]

RESEARCH_FACTS: list[FactBlueprint] = [
    FactBlueprint("citation", ("arXiv:2405.12904v3", "arXiv:2311.04412v2", "arXiv:2502.07731v1"),
                  "The key paper we must cite is {value} by Okafor et al.",
                  "Which paper must the related-work section cite?",
                  "foreground", "standalone"),
    FactBlueprint("dataset_source", ("HuggingFace:open-mem/bench-v2", "Zenodo:10.5281/zenodo.8123456", "Kaggle:longctx-2026"),
                  "The evaluation data comes from {value}, which we should not substitute.",
                  "What is the canonical source of the evaluation dataset?",
                  "background", "deep_buried"),
    FactBlueprint("date", ("2026-12-04", "2026-11-28", "2027-01-15"),
                  "The submission deadline is {value} at 17:00 UTC.",
                  "What is the submission deadline?",
                  "foreground", "embedded"),
    FactBlueprint("constraint", ("max-8k-context", "single-GPU-only", "no-proprietary-data"),
                  "The review criteria require {value}, so our method has to respect it.",
                  "What constraint do the review criteria impose?",
                  "background", "embedded"),
    FactBlueprint("version", ("v0.9.2", "v1.4.0", "v2.0.1"),
                  "We are evaluating against the {value} release of the benchmark harness.",
                  "Which release of the benchmark harness are we evaluating against?",
                  "background", "deep_buried"),
]

DAILY_FACTS: list[FactBlueprint] = [
    FactBlueprint("preference", ("tabs over spaces", "metric units only", "24-hour clock"),
                  "A standing preference: {value} in everything I send you.",
                  "What standing preference should the agent honour?",
                  "foreground", "standalone"),
    FactBlueprint("date", ("2026-11-21", "2026-12-02", "2026-11-30"),
                  "The renewal date for the subscription is {value}, please do not let it lapse.",
                  "When does the subscription renew?",
                  "foreground", "embedded"),
    FactBlueprint("preference", ("no meetings before 10am", "no calls on Fridays", "lunch at 13:00"),
                  "One more preference to respect: {value}.",
                  "Which scheduling preference did the user state?",
                  "background", "deep_buried"),
    FactBlueprint("constraint", ("budget cap 200 USD", "max two trips per quarter", "no weekend work"),
                  "There is a hard limit here: {value}.",
                  "What hard limit applies?",
                  "background", "embedded"),
    FactBlueprint("host", ("family-calendar.internal", "home-nas.local", "shared-drive.local"),
                  "The shared calendar lives on {value} and syncs nightly.",
                  "Which host serves the shared calendar?",
                  "background", "deep_buried"),
]

DOMAIN_FACTS = {"coding": CODING_FACTS, "research": RESEARCH_FACTS, "daily": DAILY_FACTS}

DOMAIN_GOALS = {
    "coding": [
        "migrate the service to the new build system",
        "cut CI time by consolidating the test matrix",
        "harden the deployment path before the release freeze",
    ],
    "research": [
        "draft the related-work section for the compression paper",
        "reproduce the baseline numbers before the deadline",
        "assemble the evaluation set from public sources",
    ],
    "daily": [
        "reorganise the weekly schedule around the new constraints",
        "set up reminders so nothing lapses this quarter",
        "consolidate the household calendar into one source",
    ],
}

# Replacement values used when a fact is superseded mid-history. Domain-scoped so
# a daily-preference fact is never "updated" into a database host. Naming is
# deliberately neutral: these become the CURRENT value, while the blueprint value
# becomes the stale one, so neither side may look "obviously old".
SUPERSEDE_VALUES: dict[str, dict[str, tuple[str, ...]]] = {
    "coding": {
        "path": ("/srv/app/.env.v2", "/opt/config/.env.current"),
        "host": ("db-stage-02.internal:5433", "staging-pg.internal:6432"),
        "version": ("v2.1.0", "v3.1.0"),
        "package": ("poetry", "pip-tools"),
        "api_key_location": ("~/.config/svc/credentials.env", "/run/secrets/svc_credentials"),
        "constraint": ("allow-new-deps", "python>=3.10"),
    },
    "research": {
        "citation": ("arXiv:2405.12904v1", "arXiv:2311.04412v1"),
        "date": ("2026-11-14", "2026-11-20"),
        "version": ("v0.8.9", "v1.3.2"),
        "dataset_source": ("HuggingFace:open-mem/bench-v1", "Zenodo:10.5281/zenodo.7000000"),
        "constraint": ("max-4k-context", "multi-GPU-only"),
    },
    "daily": {
        "date": ("2026-11-14", "2026-11-20"),
        "preference": ("spaces over tabs", "imperial units only"),
        "constraint": ("budget cap 150 USD", "max one trip per quarter"),
        "host": ("backup-calendar.internal", "shared-drive.local"),
    },
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_transcript(session: Session) -> str:
    """Canonical transcript rendering — the single source of truth.

    Both compressors and the evaluator render through this function, so the
    "full context" arm and the "compressed" arm are built from identical input.
    """
    return "".join(f"{t.role.upper()}: {t.text}\n" for t in session.turns)


def _bury(statement: str, depth: str, rng: random.Random) -> str:
    """Wrap a fact statement so its embedding depth is controlled.

    standalone   : the fact is the whole turn (easiest to keep)
    embedded     : fact sits mid-paragraph with neutral context around it
    deep_buried  : fact is inside a long assistant answer, far from any cue
    """
    if depth == "standalone":
        return statement
    if depth == "embedded":
        lead = rng.choice([
            "Here is where things stand after the last pass. ",
            "Quick note on the current configuration before we move on. ",
            "One detail worth recording, since it comes up later. ",
        ])
        tail = rng.choice([
            " Nothing else in this area changed and I do not expect it to.",
            " I will leave that as-is unless you tell me otherwise.",
            " Flag it if you want it handled differently.",
        ])
        return lead + statement + tail
    return ASSISTANT_PADDING + " " + statement + " " + ASSISTANT_PADDING


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------
@dataclass
class _Event:
    """A planned informative turn."""

    text: str
    role: str
    fact_ids: list[str] = field(default_factory=list)
    # Placement hint: "early" / "late" / "any". Used to keep the supersede
    # event late and the stale value early.
    zone: str = "any"


def _pick_domain(index: int) -> str:
    """~40% coding / 30% research / 30% daily, deterministic per index."""
    bucket = index % 10
    if bucket < 4:
        return "coding"
    if bucket < 7:
        return "research"
    return "daily"


def _build_session(index: int, seed: int, turns_range: tuple[int, int]) -> Session:
    rng = random.Random(seed * 1000 + index)
    domain = _pick_domain(index)
    lo, hi = turns_range

    session_id = f"s{index:03d}"
    blueprints = DOMAIN_FACTS[domain]
    goal = rng.choice(DOMAIN_GOALS[domain])

    # ---- choose facts -----------------------------------------------------
    n_facts = rng.randint(6, 9)
    pool = list(blueprints)
    rng.shuffle(pool)
    chosen = (pool * 3)[:n_facts]  # allow repeats only if the pool is exhausted

    facts: list[Fact] = []
    used_values: set[str] = set()
    for i, bp in enumerate(chosen):
        # Values must be unique within a session: fact attribution is done by
        # substring search, so two facts sharing a literal would make
        # "which fact survived?" unanswerable.
        candidates = [v for v in bp.values if v not in used_values] or list(bp.values)
        value = rng.choice(candidates)
        used_values.add(value)
        facts.append(Fact(
            fact_id=f"f{i + 1:02d}",
            kind=bp.kind,
            value=value,
            domain=domain,  # type: ignore[arg-type]
            first_turn=-1,
            salience=bp.salience,  # type: ignore[arg-type]
            embedding=bp.embedding,  # type: ignore[arg-type]
            stability="stable",
            critical=True,
        ))

    # Ensure a superseded fact exists even if the blueprint draw missed every
    # supersedable kind (rare, but the audit needs the axis in every session).
    domain_supersede = SUPERSEDE_VALUES.get(domain, {})
    supersedable = [f for f in facts if f.kind in domain_supersede]
    if not supersedable:
        bp = next(b for b in blueprints if b.kind in domain_supersede)
        replacement = Fact(
            fact_id=f"f{len(facts) + 1:02d}",
            kind=bp.kind,
            value=rng.choice(bp.values),
            domain=domain,  # type: ignore[arg-type]
            first_turn=-1,
            salience=bp.salience,  # type: ignore[arg-type]
            embedding=bp.embedding,  # type: ignore[arg-type]
            stability="stable",
            critical=True,
        )
        facts.append(replacement)
        chosen.append(bp)
        supersedable = [replacement]
    superseded = rng.choice(supersedable)
    if superseded is not None:
        old_value = superseded.value
        candidates = [v for v in domain_supersede[superseded.kind] if v != old_value]
        if not candidates:
            # The blueprint draw landed on the only replacement value; pick a
            # different fact of the same kind instead.
            alternates = [f for f in facts if f.kind == superseded.kind and f.value != old_value]
            if alternates:
                superseded = rng.choice(alternates)
                old_value = superseded.value
                candidates = [v for v in domain_supersede[superseded.kind] if v != old_value]
        new_value = rng.choice(candidates) if candidates else old_value + "-next"
        superseded.stability = "superseded"
        superseded.superseded_old_value = old_value
        superseded.value = new_value
        superseded.surface_forms = [new_value]
    else:
        old_value = new_value = ""

    # ---- plan informative events -----------------------------------------
    events: list[_Event] = []
    events.append(_Event(
        f"Let's get started on this: I need to {goal}. "
        f"Before we dive in, tell me what you need from me and what the first step looks like.",
        "user", [], zone="early"))
    events.append(_Event(
        f"Happy to help. To {goal} I will first map the current state, then propose the smallest "
        f"set of changes that gets us there. I will keep you posted at each checkpoint and flag "
        f"anything that looks risky before I touch it.",
        "assistant", [], zone="early"))

    for i, (fact, bp) in enumerate(zip(facts, chosen)):
        if superseded is not None and fact is superseded:
            # Early turn states the STALE value (the trap); the update comes later.
            events.append(_Event(
                _bury(bp.statement.format(value=old_value), fact.embedding, rng),
                "assistant" if fact.embedding != "standalone" else rng.choice(["user", "assistant"]),
                [],  # deliberately NOT attributed to the fact: the old value is not the truth
                zone="early"))
            continue
        text = _bury(bp.statement.format(value=fact.value), fact.embedding, rng)
        role = "user" if fact.embedding == "standalone" and rng.random() < 0.4 else "assistant"
        events.append(_Event(text, role, [fact.fact_id], zone="any"))

    # Optional second mentions for a couple of facts -> frequency 2.
    repeatable = [f for f in facts if superseded is None or f is not superseded]
    rng.shuffle(repeatable)
    for fact in repeatable[: max(1, len(repeatable) // 3)]:
        events.append(_Event(
            rng.choice([
                f"Just to keep it on record: {fact.value}.",
                f"Still the case that {fact.value} is what we use.",
                f"Re-confirming {fact.value} before we move on.",
            ]),
            "assistant", [fact.fact_id], zone="any"))

    # The update event, deliberately late.
    if superseded is not None:
        events.append(_Event(
            f"Heads up, an update: forget the old value, it is now {new_value} "
            f"(the previous one, {old_value}, is out of date).",
            "user", [superseded.fact_id], zone="late"))

    # Closing recap (high-level anchor) + routine chatter.
    recap_value = f"I have updated my notes to {new_value}. " if superseded is not None else ""
    events.append(_Event(
        f"Understood, {recap_value}To recap the objective, we are "
        f"still working to {goal}, and nothing else in the plan has changed. "
        + rng.choice(REDUNDANT_CHATTER[domain]),
        "assistant", [], zone="late"))

    # ---- fit the turn budget ---------------------------------------------
    # Structure: [early goal turns] [body: facts + repeats + filler] [late: update + recap]
    #
    # The budget is met by choosing the *amount of redundancy*, never by dropping
    # a fact: a history that is short on filler would understate the thing being
    # measured, so we pad with redundancy until the session reaches at least the
    # lower bound of the requested turn range.
    early = [e for e in events if e.zone == "early"]
    late = [e for e in events if e.zone == "late"]
    middle = [e for e in events if e.zone == "any"]

    # The body must hold every fact-bearing event AND reach the lower turn bound.
    # ``early`` events occupy body slots, so the bound is ``lo - len(late)``.
    min_body = max(len(middle), lo - len(late))
    max_body = max(min_body, hi - len(late))
    body_len = rng.randint(min_body, max_body)

    # ---- interleave -------------------------------------------------------
    body: list[_Event | None] = [None] * body_len
    for i, e in enumerate(early):
        if i < body_len:
            body[i] = e
    free_slots = list(range(len(early), body_len))
    rng.shuffle(free_slots)
    for e in middle:
        if free_slots:
            body[free_slots.pop()] = e
    leftovers = [e for e in middle if e not in body]
    body.extend(leftovers)
    body_len = len(body)

    # Fill empty body slots with redundancy, alternating speaker.
    filler_pool = list(REDUNDANT_CHATTER[domain])
    rng.shuffle(filler_pool)
    fi = 0
    for i in range(body_len):
        if body[i] is None:
            text = filler_pool[fi % len(filler_pool)]
            fi += 1
            if rng.random() < 0.22:
                text = rng.choice(USER_SMALL_TALK)
            body[i] = _Event(text, rng.choice(["user", "assistant"]), [])

    planned = [e for e in body if e is not None] + late

    # ---- guarantee the redundancy share ----------------------------------
    # The audit is only meaningful if there is real redundancy to remove. Facts
    # are mandatory, so the only lever is adding filler; this runs until the
    # redundant share reaches the target or the turn cap is hit. The cap wins:
    # exceeding the requested turn range would break the "20-30 turns" contract.
    target_redundancy = 0.52
    guard = 0
    while guard < 40:
        guard += 1
        if len(planned) >= hi:
            break
        total = sum(count_tokens(e.text) for e in planned)
        red = sum(count_tokens(e.text) for e in planned if not e.fact_ids)
        if total and red / total >= target_redundancy:
            break
        text = filler_pool[fi % len(filler_pool)]
        fi += 1
        if rng.random() < 0.22:
            text = rng.choice(USER_SMALL_TALK)
        # Insert into the body, never after the closing recap.
        insert_at = rng.randint(len(early), max(len(early), len(planned) - len(late)))
        planned.insert(insert_at, _Event(text, rng.choice(["user", "assistant"]), []))

    # ---- materialise ------------------------------------------------------
    turns: list[Turn] = [
        Turn(idx=i, role=e.role, text=e.text, fact_ids=list(e.fact_ids))  # type: ignore[arg-type]
        for i, e in enumerate(planned)
    ]

    # Repair attribution and mention lists from the materialised turns. Only the
    # CURRENT value counts as a mention; the stale value is a trap, not truth.
    for fact in facts:
        fact.mentions = [t.idx for t in turns if fact.value in t.text]
        # Rebuild fact_ids so they reflect where the current value actually is.
        for t in turns:
            if fact.value in t.text and fact.fact_id not in t.fact_ids:
                t.fact_ids.append(fact.fact_id)
        # Drop attributions for facts whose current value is NOT in the turn.
        for t in turns:
            t.fact_ids = [fid for fid in t.fact_ids
                          if any(f.fact_id == fid and f.value in t.text for f in facts)]
        fact.first_turn = min(fact.mentions) if fact.mentions else 0

    session = Session(
        session_id=session_id,
        domain=domain,  # type: ignore[arg-type]
        turns=turns,
        facts=facts,
        seed=seed,
        title=f"{domain}: {goal}",
    )
    session.full_tokens = count_tokens(render_transcript(session))
    return session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_sessions(n: int, seed: int, turns_range: tuple[int, int] = (20, 30)) -> list[Session]:
    """Build ``n`` deterministic synthetic sessions."""
    if n <= 0:
        return []
    lo, hi = turns_range
    if lo < 4 or hi < lo:
        raise ValueError(f"invalid turns_range {turns_range!r}")
    return [_build_session(i, seed, (lo, hi)) for i in range(1, n + 1)]


def save_sessions(sessions: list[Session], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_sessions(path: str) -> list[Session]:
    out: list[Session] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns = [Turn(**t) for t in d["turns"]]
            facts = []
            for fd in d["facts"]:
                fd = dict(fd)
                fd.pop("frequency", None)   # derived property, not a field
                facts.append(Fact(**fd))
            out.append(Session(
                session_id=d["session_id"], domain=d["domain"], turns=turns,
                facts=facts, seed=d.get("seed", 0), title=d.get("title", ""),
                full_tokens=d.get("full_tokens", 0),
            ))
    return out


def redundancy_stats(session: Session) -> dict:
    """Fraction of the transcript that carries no probe-relevant fact.

    This is the "should be compressed away" share. Computed over turn text so it
    is comparable to what a compressor actually sees.
    """
    total = 0
    redundant = 0
    for t in session.turns:
        n = count_tokens(t.text)
        total += n
        if not t.fact_ids:
            redundant += n
    return {
        "total_tokens": total,
        "redundant_tokens": redundant,
        "redundancy_ratio": redundant / total if total else 0.0,
    }
