"""Hierarchical LLM summarization: the "abstraction" arm of the audit.

Design rationale
----------------
OpenMemX-style memory systems keep a *gist* of long histories by recursively
summarizing them (map-reduce). That is exactly the trade we want to measure: a
summary should retain the narrative and the salient decisions while quietly
losing precise, low-frequency details (a version pin buried in a long assistant
answer, a path mentioned once). This compressor reproduces that paradigm in its
honest form — a good prompt that *explicitly* asks for the facts we later probe
for — so the audit measures the paradigm, not our prompt-writing laziness.

Pipeline
--------
1. **Chunk** the rendered transcript into ~1200-1600 token windows, splitting at
   turn boundaries wherever possible (a mid-turn cut destroys "who said what",
   which is an artifact we are auditing, not one we want to inject).
2. **Map (level 1)**: one ``client.chat("summarizer", ...)`` per chunk. The
   prompt demands verbatim preservation of paths, package names + version pins,
   API-key locations, hosts/ports/URLs, dates/deadlines, paper ids, user
   preferences, and the *latest* value of anything the session later changes.
3. **Reduce (level 2, sometimes 3)**: merge the partial notes into one final
   memory. If the merged input is itself too long it is reduced in groups first
   (a real map-reduce tree, not one oversized call), and if the final note
   overshoots the budget by more than a tolerance, one extra "compress harder"
   pass runs.
4. **Hard budget fit**: the target is ``summary_target_ratio * full_tokens``.
   Model output is never trusted to respect a token count, so the final note is
   fitted with the shared exact truncation helper and
   ``meta["truncated"]`` records that it happened.

Failure policy
--------------
The gateway is flaky (~10-20% 400s) and quota-limited. A single bad chunk must not
destroy a session's data point, so a chunk whose call still fails after the
client's own retries falls back to a truncated *raw* chunk (recorded in
``meta["fallbacks"]``). If every call fails, the compressor still returns a valid,
budgeted ``CompressionResult`` built from raw text — the run continues, and the
failure is visible in the metadata rather than silently changing the experiment.

A **circuit breaker** additionally guards the quota case: the public gateway
answers every request with ``HTTP 429 PROMOTION_MODEL_LIMIT_REACHED`` once its
free allowance is spent, and each failed call costs up to six retries with
exponential backoff. Two consecutive failures therefore open the breaker and the
rest of the session falls back immediately (visible as
``meta["circuit_breaker_open"]``). It closes again on the first success, so a
single transient 400 does not disable a session. Measured on the real gateway,
this turns a ~13-minute dead session into a sub-second fallback.

Cost accounting
---------------
``cost_tokens`` is the ledger delta around *all* summarizer calls (prompt +
completion), measured once around the whole pipeline so it cannot drift as stages
are added. Note that ``LLMClient`` caches to disk and records *zero* tokens on a
cache hit, so a warm re-run reports ``cost_tokens == 0``; the **cold-cache**
number is the authoritative one. ``meta["cache_hits"]`` distinguishes the two.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from memfaith import config
from memfaith.compression.base import (
    Compressor,
    concrete_value_score,
    fit_units_to_budget,
    measure_ledger,
    original_token_count,
    split_by_token_budget,
    target_budget,
    truncate_to_token_budget,
)
from memfaith.schema import CompressionResult, Session
from memfaith.tokens import count_tokens

# ---------------------------------------------------------------------------
# Tunables. Read from RunConfig defaults so `get_compressor(name)` needs no
# arguments from run.py; env overrides exist for ablations.
# ---------------------------------------------------------------------------
PROMPT_VERSION = "summarizer-v1"

DEFAULT_TARGET_RATIO = float(
    os.environ.get("MEMFAITH_SUMMARY_TARGET_RATIO", config.RunConfig().summary_target_ratio)
)
CHUNK_TOKEN_BUDGET = int(os.environ.get("MEMFAITH_SUMMARY_CHUNK_TOKENS", "1400"))
# Reasoning models burn hidden tokens before emitting content; CONTRACT.md says
# >= 900 for the summarizer role and we honour that.
CHUNK_MAX_TOKENS = int(os.environ.get("MEMFAITH_SUMMARY_CHUNK_MAX_TOKENS", "1100"))
REDUCE_MAX_TOKENS = int(os.environ.get("MEMFAITH_SUMMARY_REDUCE_MAX_TOKENS", "1100"))
# Cap for the *input* of a single reduce call; above this we group-reduce first.
REDUCE_INPUT_BUDGET = int(os.environ.get("MEMFAITH_SUMMARY_REDUCE_INPUT_TOKENS", "3200"))
# Overshoot tolerated before spending an extra "compress harder" pass.
OVERSHOOT_TOLERANCE = 1.10
# Below this fraction of the budget, the note is considered under-filled and the
# leftover budget is spent on verbatim, fact-dense original lines (recorded in
# meta["backfilled"]). Without this, a terse summarizer would "win" the accuracy
# comparison purely by carrying less context than the extractive arm.
MIN_BUDGET_UTILIZATION = float(os.environ.get("MEMFAITH_SUMMARY_MIN_UTILIZATION", "0.75"))

_SYSTEM_PROMPT = """You compress a slice of an agent's conversation history into a compact memory note. A later agent will answer factual questions using ONLY your note.

You are not writing a narrative. You are writing a retrieval key.

NON-NEGOTIABLE PRESERVATION RULES — copy these VERBATIM, character for character, even if they look like noise:
1. File paths and directories (e.g. /srv/app/config/prod.env, src/api/handlers.py).
2. Package/library names with their exact version pins (e.g. pydantic==2.14.3, numpy 2.5.4).
3. Where secrets / API keys live: the exact path or secret-store name (never the secret value).
4. Hosts, ports, URLs and registry endpoints (e.g. registry.internal:8443, https://...).
5. Dates, deadlines, time windows and schedules.
6. Paper titles, arXiv ids, DOIs, citation keys and dataset names.
7. User preferences, constraints and explicit instructions ("always use pnpm", "never touch prod").
8. Decisions taken, and the rejected alternative when both were stated.

UPDATE SEMANTICS: when the conversation changes a value (a newer version pin, a moved path, a corrected host), keep the LATEST value and mark the earlier one as superseded in at most six words. Never present a stale value as current.

COMPRESSION RULES:
- Output compact bullet points, one fact per bullet. No preamble, no closing remarks, no headings.
- Prefix every bullet with the speaker who established the fact: USER or ASSISTANT.
- Drop greetings, acknowledgements, re-stated build steps, status chatter and anything repeated. Removing redundancy is the main job.
- Never invent details. If the input does not say it, your note does not say it.
- Never replace a concrete value with a placeholder such as "..." or "various".
- Keep the note under {target_tokens} tokens.

GOAL: the smallest note that still lets another agent answer "which exact path / package / version / host / date / paper / preference was settled on?" without seeing the original conversation."""

_REDUCE_SYSTEM_PROMPT = """You merge several partial memory notes about ONE agent session into a single final memory note.

The notes were written independently from consecutive slices of the same conversation, so they overlap. Merge them; do not concatenate them.

MERGE RULES:
- Deduplicate aggressively: the same fact stated in several notes appears ONCE.
- Keep every concrete detail verbatim: file paths, package names with version pins, secret/key locations, hosts, ports, URLs, dates and deadlines, paper ids, dataset names, user preferences and constraints.
- For values that were later changed, keep only the LATEST value and, at most, a six-word note that an earlier one was superseded.
- Preserve speaker attribution: prefix each bullet with USER or ASSISTANT.
- Order bullets by topic, not by which slice they came from.
- Output compact bullets only. No preamble, no headings, no closing remarks.
- Hard limit: the final note must be under {target_tokens} tokens. Prefer dropping a repeated or trivial detail over dropping a concrete value."""

_HARDER_SYSTEM_PROMPT = """The memory note below is too long. Compress it further to fit a hard token budget.

Rules:
- Keep every concrete value: paths, package names and version pins, key/secret locations, hosts, ports, URLs, dates, paper ids, dataset names, user preferences.
- Drop reasoning, connective prose, restated context and any bullet that contains no concrete value.
- Merge bullets about the same topic.
- Output bullets only. No preamble, no headings.
- Hard limit: under {target_tokens} tokens."""

_EXPAND_SYSTEM_PROMPT = """Your memory note is too thin. Another agent must answer factual questions about this session using only your note, and right now it is missing details it has room for.

Rules:
- Keep every fact already in the note.
- Add the concrete values from the conversation that the note omits: file paths, package names with version pins, secret/key locations, hosts, ports, URLs, dates and deadlines, paper ids, dataset names, user preferences and constraints, and decisions with the alternative that was rejected.
- For values that changed, state the LATEST value and note in at most six words that an earlier one was superseded.
- Prefix every bullet with the speaker who established the fact: USER or ASSISTANT.
- Bullets only. No preamble, no headings, no closing remarks.
- Never invent details that the conversation does not contain.
- Hard limit: under {target_tokens} tokens."""


@dataclass(frozen=True)
class _Chunk:
    """A window of the transcript, aligned to turn boundaries where possible."""

    index: int
    text: str
    first_turn: int
    last_turn: int
    tokens: int


def _render_turn(role: str, text: str) -> str:
    return f"{role.upper()}: {text}\n"


def chunk_transcript(session: Session, budget: int = CHUNK_TOKEN_BUDGET) -> list[_Chunk]:
    """Split a session into ~``budget``-token windows without cutting turns.

    A turn longer than the whole budget is split (unavoidable), with the role
    label repeated on each piece so every fragment stays attributable.
    """
    budget = max(64, int(budget))
    chunks: list[_Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    first_turn = -1
    last_turn = -1

    def flush() -> None:
        nonlocal buf, buf_tokens, first_turn, last_turn
        if not buf:
            return
        text = "".join(buf)
        chunks.append(_Chunk(len(chunks), text, first_turn, last_turn, count_tokens(text)))
        buf, buf_tokens, first_turn, last_turn = [], 0, -1, -1

    for turn in session.turns:
        line = _render_turn(turn.role, turn.text)
        line_tokens = count_tokens(line)

        if line_tokens > budget:
            # Oversized single turn: emit what we have, then split the turn.
            flush()
            pieces = split_by_token_budget(line, budget)
            for piece in pieces:
                chunks.append(_Chunk(len(chunks), piece, turn.idx, turn.idx, count_tokens(piece)))
            continue

        if buf and buf_tokens + line_tokens > budget:
            flush()
        if not buf:
            first_turn = turn.idx
        buf.append(line)
        buf_tokens += line_tokens
        last_turn = turn.idx

    flush()
    if not chunks:
        chunks.append(_Chunk(0, "", 0, -1, 0))
    return chunks


def _group_by_budget(texts: list[str], budget: int) -> list[list[str]]:
    """Greedily group texts so each group's joined size stays near ``budget``."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for text in texts:
        n = count_tokens(text)
        if cur and cur_tokens + n > budget:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(text)
        cur_tokens += n
    if cur:
        groups.append(cur)
    return groups or [[]]


class HierarchicalSummaryCompressor(Compressor):
    """Map-reduce summarizer: keep the gist, audit what specifics survive."""

    name = "summary_hierarchical"

    def __init__(
        self,
        target_ratio: float | None = None,
        chunk_token_budget: int | None = None,
        chunk_max_tokens: int | None = None,
        reduce_input_budget: int | None = None,
    ):
        self.target_ratio = DEFAULT_TARGET_RATIO if target_ratio is None else float(target_ratio)
        self.chunk_token_budget = CHUNK_TOKEN_BUDGET if chunk_token_budget is None else int(chunk_token_budget)
        self.chunk_max_tokens = CHUNK_MAX_TOKENS if chunk_max_tokens is None else int(chunk_max_tokens)
        self.reduce_input_budget = (
            REDUCE_INPUT_BUDGET if reduce_input_budget is None else int(reduce_input_budget)
        )

    # -- prompt construction ------------------------------------------------
    @staticmethod
    def _map_messages(chunk: _Chunk, total: int) -> list[dict]:
        span = (
            f"turns {chunk.first_turn}-{chunk.last_turn}"
            if chunk.last_turn >= chunk.first_turn >= 0
            else "unknown turns"
        )
        target = max(120, min(CHUNK_TOKEN_BUDGET // 3, 480))
        system = _SYSTEM_PROMPT.format(target_tokens=target)
        user = (
            f"Conversation slice {chunk.index + 1} of {total} ({span}).\n"
            f"Compress it into one memory note under {target} tokens.\n\n"
            f"--- BEGIN SLICE ---\n{chunk.text}--- END SLICE ---"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _reduce_messages(notes: list[str], target_tokens: int) -> list[dict]:
        system = _REDUCE_SYSTEM_PROMPT.format(target_tokens=target_tokens)
        body = "\n\n".join(f"--- NOTE {i + 1} ---\n{n}" for i, n in enumerate(notes))
        user = (
            f"Merge the {len(notes)} partial notes below into ONE final memory note "
            f"under {target_tokens} tokens.\n\n{body}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _harder_messages(note: str, target_tokens: int) -> list[dict]:
        system = _HARDER_SYSTEM_PROMPT.format(target_tokens=target_tokens)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compress this note under {target_tokens} tokens:\n\n{note}"},
        ]

    # -- main ---------------------------------------------------------------
    def compress(self, session: Session, client) -> CompressionResult:
        original_tokens = original_token_count(session)
        budget = target_budget(original_tokens, self.target_ratio)

        meta: dict = {
            "prompt_version": PROMPT_VERSION,
            "model": getattr(client, "models", {}).get("summarizer") if client is not None else None,
            "levels": 0,
            "chunks": 0,
            "truncated": False,
            "fallbacks": [],
            "expanded": False,
            "backfilled": 0,
            "target_ratio": self.target_ratio,
            "budget_tokens": budget,
        }

        # ONE ledger measurement around every LLM call this compress() makes, so
        # cost_tokens is exact by construction and cannot drift when a stage is
        # added. (Earlier a nested measurement double-counted the expand pass.)
        with measure_ledger(client) as ledger_delta:
            text, levels, fallbacks, breaker = self._build_memory(session, client, budget, meta)
            meta["levels"] = levels
            meta["fallbacks"] = fallbacks
            if breaker.opened_by:
                meta["circuit_breaker_open"] = breaker.opened_by

            # Hard budget fit. Model output is never trusted to respect a token
            # count; this is the step that makes the achieved ratio exact.
            # Whole bullets are kept where possible: a half-cut bullet
            # (".../srv/app/secr") is not a fact, it is noise that still costs
            # tokens. The drop order is importance-based, not positional, so a
            # budget cut cannot by itself erase the newest value of an updated
            # fact (see ``_bullet_priority``).
            text, truncated, kept_units, total_units = self._fit(text, budget)
            meta["truncated"] = bool(truncated or meta.get("truncated"))
            meta["bullets_kept"] = kept_units
            meta["bullets_total"] = total_units

            # Under-fill handling. A terse summarizer can return far less than
            # the budget allows, and the two arms would then not be compared at
            # equal context size — the summary arm would "win" on accuracy by
            # simply carrying less text. Padding with raw transcript would
            # corrupt the paradigm, so the first remedy is in-paradigm: ask the
            # summarizer for a richer note at the budget it was given. Only if it
            # is *still* short do we spend the leftover budget on the most
            # fact-dense original lines (verbatim). Both remedies are recorded in
            # meta so the metrics layer can see how much of the arm's output is
            # abstracted versus borrowed.
            if (
                budget > 0
                and count_tokens(text) < int(budget * MIN_BUDGET_UTILIZATION)
                and breaker.closed
            ):
                expanded = self._expand(client, session, text, budget, meta, fallbacks)
                if expanded:
                    text, truncated, kept_units, total_units = self._fit(expanded, budget)
                    meta["truncated"] = bool(truncated or meta.get("truncated"))
                    meta["bullets_kept"] = kept_units
                    meta["bullets_total"] = total_units

            if budget > 0 and count_tokens(text) < int(budget * MIN_BUDGET_UTILIZATION):
                text, backfill_lines = self._backfill(session, text, budget)
                meta["backfilled"] = len(backfill_lines)
            else:
                meta["backfilled"] = 0

        delta = ledger_delta()
        meta.update(delta.to_meta())

        compressed_tokens = count_tokens(text)
        meta["budget_utilization"] = round(compressed_tokens / budget, 4) if budget else 0.0
        ratio = (compressed_tokens / original_tokens) if original_tokens else 0.0

        return CompressionResult(
            session_id=session.session_id,
            strategy=self.name,
            text=text,
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            ratio=ratio,
            cost_tokens=delta.cost_tokens,
            meta=meta,
        )

    # -- internals ----------------------------------------------------------
    def _fit(self, text: str, budget: int) -> tuple[str, bool, int, int]:
        """Hard-fit ``text`` to ``budget`` by dropping whole, least-valuable bullets.

        Returns ``(fitted_text, truncated, bullets_kept, bullets_total)``.
        """
        units = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        fitted, truncated, kept_units = fit_units_to_budget(
            units, budget, marker=" …", priority=self._bullet_priority(units)
        )
        return fitted, truncated, kept_units, len(units)

    def _expand(
        self,
        client,
        session: Session,
        note: str,
        budget: int,
        meta: dict,
        fallbacks: list[dict],
    ) -> str:
        """Ask the summarizer for a fuller note when the first pass was too terse.

        This stays inside the abstraction paradigm: the model is shown the
        transcript slices and its own thin note, and asked to add the concrete
        details it left out, up to the budget it was given. Returns the new note,
        or ``""`` when the call fails or produces nothing longer (in which case
        the caller falls back to verbatim backfill).
        """
        if not note.strip() or client is None:
            meta["expanded"] = False
            return ""
        chunks = chunk_transcript(session, self.chunk_token_budget)
        source = "\n".join(c.text for c in chunks)[: self.chunk_token_budget * 3]
        target = max(64, int(budget * 0.95))
        messages = [
            {"role": "system", "content": _EXPAND_SYSTEM_PROMPT.format(target_tokens=target)},
            {
                "role": "user",
                "content": (
                    f"Your current memory note is too thin for its {target}-token budget.\n\n"
                    f"--- CURRENT NOTE ---\n{note}\n--- END NOTE ---\n\n"
                    f"--- ORIGINAL CONVERSATION (truncated) ---\n{source}\n--- END CONVERSATION ---\n\n"
                    f"Rewrite the note, keeping every fact already present and adding the concrete "
                    f"values from the conversation that it omits. Stay under {target} tokens."
                ),
            },
        ]
        try:
            out = client.chat(
                "summarizer", messages, max_tokens=REDUCE_MAX_TOKENS,
                tag=f"compress:expand:{session.session_id}",
            )
        except Exception as e:  # noqa: BLE001
            fallbacks.append({"stage": "expand", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            meta["expanded"] = False
            return ""
        out = (out or "").strip()
        meta["expanded"] = bool(out)
        return out if count_tokens(out) > count_tokens(note) else ""
    def _backfill(self, session: Session, text: str, budget: int) -> tuple[str, list[str]]:
        """Spend unused budget on the most fact-dense original lines, verbatim.

        Only used when the summarizer is materially under budget (see
        ``MIN_BUDGET_UTILIZATION``). The added lines are verbatim transcript with
        their speaker label, chosen by the same concrete-value score the
        extractive arm uses, so the summary arm reaches a comparable context size
        without being handed a *better* notion of value than the extractive arm.
        """
        if budget <= 0:
            return text, []
        remaining = budget - count_tokens(text)
        if remaining <= 8:
            return text, []

        candidates: list[tuple[float, int, str]] = []
        for turn in session.turns:
            for line in turn.text.splitlines():
                stripped = line.strip()
                if len(stripped) < 12:
                    continue
                rendered = f"{turn.role.upper()}: {stripped}"
                candidates.append((concrete_value_score(stripped), turn.idx, rendered))
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        if not candidates:
            return text, []

        # Only genuinely informative lines qualify; never pad with chatter.
        top_score = candidates[0][0]
        if top_score <= 0.0:
            return text, []

        added: list[str] = []
        for score, _turn_idx, rendered in candidates[:60]:
            if score <= 0.0:
                break
            trial = "\n".join([text, *added, rendered]) if text else "\n".join([*added, rendered])
            if count_tokens(trial) > budget:
                continue
            added.append(rendered)
            if count_tokens(trial) >= budget:
                break
        if not added:
            return text, []
        return (text + "\n" + "\n".join(added) if text else "\n".join(added)), added

    @staticmethod
    def _bullet_priority(units: list[str]) -> list[int]:
        """Rank bullets by how much concrete information they carry.

        Used only when the note must be hard-cut to budget. Ranking by concrete
        values (paths, versions, hosts, ids, dates, cues) rather than by position
        keeps the cut from systematically deleting the end of the session — and
        the end is where superseded facts hold their *current* value. Ties go to
        the *later* bullet for the same reason: when two equally informative
        bullets compete and only one fits, the more recent statement is the one
        that is still true.
        """
        scored = [(concrete_value_score(u), i) for i, u in enumerate(units)]
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [i for _score, i in scored]

    def _build_memory(
        self, session: Session, client, budget: int, meta: dict
    ) -> tuple[str, int, list[dict], "_CircuitBreaker"]:
        """Run the map-reduce tree. Returns ``(text, levels, fallbacks, breaker)``."""
        chunks = chunk_transcript(session, self.chunk_token_budget)
        meta["chunks"] = len(chunks)
        fallbacks: list[dict] = []
        notes: list[str] = []
        breaker = _CircuitBreaker()

        # ---- level 1: map ------------------------------------------------
        for chunk in chunks:
            note = ""
            if client is not None and breaker.closed:
                try:
                    note = client.chat(
                        "summarizer",
                        self._map_messages(chunk, len(chunks)),
                        max_tokens=self.chunk_max_tokens,
                        tag=f"compress:map:{session.session_id}:{chunk.index}",
                    )
                    breaker.record_success()
                except Exception as e:  # noqa: BLE001 - resilience is the point
                    note = ""
                    breaker.record_failure(e)
                    fallbacks.append(
                        {
                            "stage": "map",
                            "chunk": chunk.index,
                            "turns": [chunk.first_turn, chunk.last_turn],
                            "error": f"{type(e).__name__}: {str(e)[:200]}",
                        }
                    )
            elif client is not None:
                fallbacks.append(
                    {"stage": "map", "chunk": chunk.index, "error": "circuit breaker open"}
                )
            else:
                fallbacks.append({"stage": "map", "chunk": chunk.index, "error": "no client"})

            note = (note or "").strip()
            if not note:
                # Fallback: keep the raw slice, budget-fitted, so the final
                # reduce still sees the original wording of this window.
                raw_budget = max(120, min(chunk.tokens, self.chunk_token_budget // 2))
                note, _ = truncate_to_token_budget(chunk.text.strip(), raw_budget, marker=" …")
            notes.append(note)

        if not notes:
            return "", 1, fallbacks, breaker

        # ---- level 2: reduce (grouped if the merged input is too big) -----
        levels = 1
        target_tokens = max(64, budget)
        joined = "\n\n".join(notes)
        if count_tokens(joined) > self.reduce_input_budget:
            groups = _group_by_budget(notes, self.reduce_input_budget)
            group_target = max(128, target_tokens // len(groups))
            group_notes: list[str] = []
            for gi, group in enumerate(groups):
                group_notes.append(
                    self._chat_or_join(
                        client,
                        self._reduce_messages(group, group_target),
                        fallback="\n".join(group),
                        session_id=session.session_id,
                        stage=f"reduce-group-{gi}",
                        fallbacks=fallbacks,
                        breaker=breaker,
                    )
                )
            levels += 1
            notes = group_notes

        final = self._chat_or_join(
            client,
            self._reduce_messages(notes, target_tokens),
            fallback="\n\n".join(notes),
            session_id=session.session_id,
            stage="reduce-final",
            fallbacks=fallbacks,
            breaker=breaker,
        )
        levels += 1

        # ---- extra pass if the model blew the budget ----------------------
        if count_tokens(final) > int(target_tokens * OVERSHOOT_TOLERANCE):
            final = self._chat_or_join(
                client,
                self._harder_messages(final, target_tokens),
                fallback=final,
                session_id=session.session_id,
                stage="reduce-extra",
                fallbacks=fallbacks,
                breaker=breaker,
            )
            levels += 1

        return final.strip(), levels, fallbacks, breaker

    def _chat_or_join(
        self,
        client,
        messages: list[dict],
        *,
        fallback: str,
        session_id: str,
        stage: str,
        fallbacks: list[dict],
        breaker: _CircuitBreaker | None = None,
    ) -> str:
        """Call the summarizer; on failure return ``fallback`` and record why."""
        if client is None:
            fallbacks.append({"stage": stage, "error": "no client"})
            return fallback
        if breaker is not None and breaker.open:
            fallbacks.append({"stage": stage, "error": "circuit breaker open"})
            return fallback
        try:
            out = client.chat(
                "summarizer",
                messages,
                max_tokens=REDUCE_MAX_TOKENS,
                tag=f"compress:{stage}:{session_id}",
            )
            if breaker is not None:
                breaker.record_success()
        except Exception as e:  # noqa: BLE001
            if breaker is not None:
                breaker.record_failure(e)
            fallbacks.append({"stage": stage, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return fallback
        out = (out or "").strip()
        return out or fallback


class _CircuitBreaker:
    """Stop hammering a dead gateway mid-session.

    The public gateway is quota-limited and returns ``HTTP 429
    PROMOTION_MODEL_LIMIT_REACHED`` for *every* call once the free allowance is
    spent, and each failed call costs the client up to 6 retries with exponential
    backoff. Without this, a quota-exhausted run spends ~13 minutes per session
    doing nothing and produces a results file that looks like a compression
    finding rather than an outage.

    After ``max_consecutive_failures`` consecutive failures the breaker opens and
    the remaining chunks/reduces fall back to raw text immediately. It closes
    again on the first success, so a single transient 400 does not disable the
    session. The opener state is visible in ``meta["fallbacks"]``.
    """

    __slots__ = ("max_consecutive_failures", "_consecutive", "opened_by")

    def __init__(self, max_consecutive_failures: int = 2):
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive = 0
        self.opened_by = ""

    @property
    def open(self) -> bool:
        return self._consecutive >= self.max_consecutive_failures

    @property
    def closed(self) -> bool:
        return not self.open

    def record_success(self) -> None:
        self._consecutive = 0
        self.opened_by = ""

    def record_failure(self, exc: BaseException) -> None:
        self._consecutive += 1
        if self._consecutive == self.max_consecutive_failures:
            self.opened_by = f"{type(exc).__name__}: {str(exc)[:200]}"


__all__ = [
    "CHUNK_TOKEN_BUDGET",
    "DEFAULT_TARGET_RATIO",
    "HierarchicalSummaryCompressor",
    "MIN_BUDGET_UTILIZATION",
    "PROMPT_VERSION",
    "chunk_transcript",
]
