"""Extractive sentence selection: the "verbatim" arm of the audit.

Design rationale
----------------
AgentMem-style layered memories and LLMLingua-style prompt compression both
assume that *selecting* existing text is safer than *rewriting* it: nothing is
paraphrased, so a path or a version pin cannot be silently mangled. The cost is
context: a sentence lifted out of a long answer loses the reasoning around it,
and a stale value can survive while the sentence that superseded it is dropped.

This compressor is deliberately model-free: no network, no downloads, fully
deterministic, so it is a clean control for "how much fidelity do you buy with
zero LLM spend?". It implements a small TF-IDF / MMR-style scorer:

1. **Segmentation** keeps the transcript's speaker structure. Each sentence is
   tagged with its turn and role, and the rendered memory repeats ``USER:`` /
   ``ASSISTANT:`` prefixes, so extracted text stays attributable — an audit of
   "who said what" would otherwise be measuring our own formatting.
2. **Term saliency**: TF-IDF over the session's own sentences. This is what
   removes the redundancy the dataset injects (status chatter, re-stated build
   steps): a phrase repeated in ten turns has a low IDF and therefore a low
   score.
3. **High-value boosters**: regexes for the fact kinds the probes care about —
   paths, URLs, dotted identifiers, version pins, dates/deadlines, arXiv ids,
   ports, hosts, quoted literals, ALLCAPS keywords, secret locations and
   remember/important cues. Boosters are additive and large enough to lift a
   short factual sentence above a long chatty one.
4. **Redundancy penalty (MMR)**: a candidate that is nearly identical to an
   already-selected sentence is penalised, and near-duplicates are skipped
   outright, so the budget goes to new information.
5. **Recency for superseded facts**: if two sentences mention the same concrete
   entity (same path / version / host / quoted value) the *later* one is boosted
   and the earlier one damped. Probes ask for the current value, so an
   extractive memory that keeps the old value would fail for a reason that has
   nothing to do with token budget.
6. **Budget**: ``extractive_target_ratio * full_tokens``, enforced exactly with
   the shared truncation helper. Selection is greedy by score-per-token, then a
   bounded fill pass adds lower-ranked sentences while the *rendered* text still
   fits, so the budget is used but never exceeded.

The optional LLMLingua-2 adapter at the bottom is opt-in only
(``MEMFAITH_USE_LLMLINGUA=1`` plus an already-local model); it never downloads
anything and falls back to this compressor if unavailable.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field

from memfaith import config
from memfaith.compression.base import (
    CONCRETE_VALUE_PATTERNS,
    Compressor,
    concrete_entities,
    concrete_value_score,
    measure_ledger,
    original_token_count,
    render_transcript,
    stale_and_latest_positions,
    target_budget,
    truncate_to_token_budget,
)
from memfaith.schema import CompressionResult, Session
from memfaith.tokens import count_tokens

log = logging.getLogger(__name__)

METHOD = "tfidf+entity-boost+mmr-greedy-v1"
DEFAULT_TARGET_RATIO = float(
    os.environ.get("MEMFAITH_EXTRACTIVE_TARGET_RATIO", config.RunConfig().extractive_target_ratio)
)

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
_ABBREV = frozenset({"e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "no.", "fig.", "al.", "mr.", "dr."})
# A sentence boundary: terminal punctuation (with optional closing quotes) then space.
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*\s+")
_VERSION_TAIL_RE = re.compile(r"\d+(?:\.\d+)+(?:\.?)$")
_URLISH_RE = re.compile(r"(?:https?://|www\.)\S+$", re.IGNORECASE)


def _split_line(line: str) -> list[str]:
    """Split one physical line into sentences, keeping code/identifier runs intact.

    The guard rails matter: naive ``split(". ")`` shreds ``pydantic==2.14.3``,
    URLs and ``e.g.`` into fragments that then lose their value.
    """
    pieces: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(line):
        end = m.start() + len(m.group(0).rstrip())
        left = line[start:end]
        right = line[m.end():]
        if not left.strip():
            continue
        low = left.strip().lower()
        if low.endswith(tuple(_ABBREV)):
            continue
        # Don't cut inside a version pin / dotted identifier / URL.
        if _VERSION_TAIL_RE.search(left.strip()) and right[:1].islower():
            continue
        if _URLISH_RE.search(left.strip()) and not left.rstrip().endswith(("!", "?")):
            continue
        # A boundary before a lowercase word is almost always mid-sentence.
        if right[:1].islower() and not left.rstrip().endswith(("!", "?")):
            continue
        pieces.append(left.strip())
        start = m.end()
    tail = line[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


def split_sentences(text: str) -> list[str]:
    """Split a turn body into sentence/bullet units (newlines are hard units)."""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line:
            out.extend(_split_line(line))
    return out


@dataclass
class _Sentence:
    """One selectable unit, with its provenance and derived features."""

    order: int                 # global ordinal (original document order)
    turn_idx: int
    role: str
    text: str
    tokens: int                # cost when rendered alone as "ROLE: text\n"
    terms: frozenset[str] = field(default_factory=frozenset)
    entities: frozenset[str] = field(default_factory=frozenset)
    base_score: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']{1,}|[0-9][0-9.]*")
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those i you we they he she it
    is are was were be been being do does did done have has had can could should would
    will may might must not no yes ok okay so just now also very really quite maybe
    of to in on at for from with by as about into over under again more most some any
    all each other such only own same too s t don't doesn't didn't i'm i've we're
    let's get got go going want need like know think see say said tell told
    """.split()
)

# The concrete-value model lives in ``base`` so the summary arm ranks bullets with
# exactly the same notion of "valuable" that this arm scores with.
_BOOSTERS = CONCRETE_VALUE_PATTERNS


def _terms(text: str) -> frozenset[str]:
    """Content terms for TF-IDF: lowercase words, dropping stopwords and 1-char tokens."""
    out: set[str] = set()
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if len(w) < 2 or w in _STOPWORDS:
            continue
        out.add(w)
    return frozenset(out)


def _booster_score(text: str) -> float:
    """Concrete-value score of one sentence (paths, versions, hosts, ids, cues)."""
    return concrete_value_score(text)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------
class ExtractiveCompressor(Compressor):
    """Deterministic, model-free sentence selection under a token budget."""

    name = "extractive"

    def __init__(
        self,
        target_ratio: float | None = None,
        redundancy_threshold: float = 0.72,
        near_duplicate_threshold: float = 0.85,
    ):
        self.target_ratio = DEFAULT_TARGET_RATIO if target_ratio is None else float(target_ratio)
        self.redundancy_threshold = redundancy_threshold
        self.near_duplicate_threshold = near_duplicate_threshold

    # -- scoring ------------------------------------------------------------
    def _score(self, sentences: list[_Sentence]) -> None:
        n = len(sentences)
        if not n:
            return

        # Document frequency over sentences -> IDF. Redundant chatter repeats
        # the same terms, so its IDF (and score) collapses.
        df: dict[str, int] = {}
        for s in sentences:
            for term in s.terms:
                df[term] = df.get(term, 0) + 1
        idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        # Supersession pass, shared with the summary arm: which sentences hold a
        # value that a later sentence replaced, and which hold the newest value
        # of an updated slot. Entities are already extracted, so reuse them.
        stale_orders, newest_orders = stale_and_latest_positions([s.entities for s in sentences])

        max_turn = max(s.turn_idx for s in sentences) or 1
        for s in sentences:
            if not s.terms:
                s.base_score = 0.0
                continue
            tf = len(s.terms) / (len(s.terms) + 6.0)          # length-normalised term mass
            salience = sum(idf.get(t, 1.0) for t in s.terms) / len(s.terms)
            core = 4.0 * tf * salience
            boosted = core + _booster_score(s.text)

            # Last-mention weighting: the sentence holding the newest value of a
            # slot outranks the one holding a value that was later replaced.
            # Probes ask for the CURRENT value, so keeping the stale one would
            # fail for a reason unrelated to the token budget.
            if s.order in stale_orders:
                boosted *= 0.45
            elif s.order in newest_orders:
                boosted *= 1.35

            # Very mild recency: later turns are marginally more current, but
            # this must never dominate a concrete-value booster.
            boosted *= 1.0 + 0.06 * (s.turn_idx / max_turn)

            s.base_score = boosted

    # -- selection ----------------------------------------------------------
    def _select(self, sentences: list[_Sentence], budget: int) -> list[_Sentence]:
        """Greedy selection under ``budget``, ordered by value per token spent.

        Ranking by raw score would let one long, chatty sentence consume the
        budget a short fact-carrying sentence could have used. Ranking by
        ``score / tokens**0.75`` is a soft length normalisation: it prefers
        dense, fact-bearing sentences without turning the arm into a "keep the
        shortest sentences" heuristic (the exponent < 1 still rewards a longer
        sentence when it carries more concrete values).

        Redundancy is checked exactly (max Jaccard against everything selected so
        far) but only against sentences that share at least one term, via an
        inverted index. A naive all-pairs loop is O(n^2) and becomes the
        bottleneck on a 400-turn history; the index makes the common case (a
        candidate unrelated to the selection) a no-op.
        """
        order = sorted(
            sentences,
            key=lambda s: (-(s.base_score / (max(1, s.tokens) ** 0.75)), s.order),
        )
        selected: list[_Sentence] = []
        used = 0
        # term -> positions in ``selected`` that contain it
        index: dict[str, list[int]] = {}

        for cand in order:
            if cand.base_score <= 0.0:
                continue
            if cand.tokens > budget:
                continue
            if used + cand.tokens > budget:
                continue

            # Only compare with selected sentences sharing a term.
            peers: set[int] = set()
            for term in cand.terms:
                postings = index.get(term)
                if postings:
                    peers.update(postings)

            overlap = 0.0
            duplicate = False
            for j in peers:
                sim = _jaccard(cand.terms, selected[j].terms)
                if sim > overlap:
                    overlap = sim
                if sim >= self.near_duplicate_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            # MMR penalty: skip candidates that largely repeat what we already
            # have, unless they carry a strong concrete-value booster.
            if overlap >= self.redundancy_threshold and cand.base_score < 6.0:
                continue

            pos = len(selected)
            selected.append(cand)
            used += cand.tokens
            for term in cand.terms:
                index.setdefault(term, []).append(pos)

        return selected

    @staticmethod
    def _render(selected: list[_Sentence]) -> str:
        """Render in ORIGINAL order, grouped per turn, with speaker labels kept."""
        if not selected:
            return ""
        lines: list[str] = []
        cur_turn: int | None = None
        cur_role = ""
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            if buf:
                lines.append(f"{cur_role.upper()}: {' '.join(buf)}")
                buf = []

        for s in sorted(selected, key=lambda x: x.order):
            if s.turn_idx != cur_turn:
                flush()
                cur_turn, cur_role = s.turn_idx, s.role
            buf.append(s.text)
        flush()
        return "\n".join(lines)

    def compress(self, session: Session, client=None) -> CompressionResult:
        """Select and render a budgeted subset. ``client`` is unused (no LLM)."""
        original_tokens = original_token_count(session)
        budget = target_budget(original_tokens, self.target_ratio)

        # The ledger is untouched by design; measuring anyway keeps the cost
        # contract uniform across arms and future-proofs a learned selector.
        with measure_ledger(client) as ledger_delta:
            sentences = self._build_sentences(session)
            self._score(sentences)
            selected = self._select(sentences, budget)
            text = self._render(selected)
            text, kept = self._fill(text, selected, sentences, budget)

            truncated = False
            if count_tokens(text) > budget:  # defensive: _render/_fill should fit
                text, truncated = truncate_to_token_budget(text, budget, marker=" …")

            compressed_tokens = count_tokens(text)

        delta = ledger_delta()

        meta = {
            "method": METHOD,
            "kept_sentences": kept,
            "total_sentences": len(sentences),
            "budget_tokens": budget,
            "target_ratio": self.target_ratio,
            "llmlingua": False,
            "truncated": truncated,
        }
        meta.update(delta.to_meta())
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

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _build_sentences(session: Session) -> list[_Sentence]:
        out: list[_Sentence] = []
        for turn in session.turns:
            for sent in split_sentences(turn.text):
                prefix = f"{turn.role.upper()}: "
                out.append(
                    _Sentence(
                        order=len(out),
                        turn_idx=turn.idx,
                        role=turn.role,
                        text=sent,
                        tokens=count_tokens(prefix + sent + "\n"),
                        terms=_terms(sent),
                        entities=concrete_entities(sent),
                    )
                )
        return out

    def _fill(
        self, text: str, selected: list[_Sentence], sentences: list[_Sentence], budget: int
    ) -> tuple[str, int]:
        """Add next-best unselected sentences while the *exact* render still fits.

        Selection is greedy on score, which leaves a ragged tail of budget. This
        pass fills it with the highest-scoring remaining sentences so the arm
        actually uses its budget (an under-filled arm would "win" accuracy by
        carrying less context).

        Cost note: re-rendering the whole selection per candidate is O(n^2) in
        characters — fine for a 20-30 turn session, too slow on a 400-turn one.
        So candidates are triaged with a *per-sentence* cost estimate (its own
        rendered length) and only the survivors are verified against the exact
        rendered count. The final text is re-checked, and if the estimate was
        optimistic the last additions are dropped until it fits — the budget
        guarantee therefore never depends on the estimate being right.
        """
        chosen = {s.order for s in selected}
        current = count_tokens(text)
        if current >= budget:
            return text, len(selected)

        pool = sorted(
            (s for s in sentences if s.order not in chosen and s.base_score > 0.0),
            key=lambda s: (-(s.base_score / (max(1, s.tokens) ** 0.75)), s.order),
        )
        if not pool:
            return text, len(selected)
        cheapest = min(s.tokens for s in pool)

        # Rendering groups sentences per turn, so appending a sentence to a turn
        # that already has text costs " text" while starting a new turn costs
        # "\nROLE: text". Track those per-turn buffers so each admission decision
        # is O(sentence), not O(whole output) — the difference between ~20 ms and
        # ~2 s on a 400-turn history. The running total is re-verified against the
        # exact rendered text periodically and at the end, so the budget
        # guarantee never rests on the incremental arithmetic.
        turn_lines: dict[int, list[str]] = {}
        turn_roles: dict[int, str] = {}
        for s in selected:
            turn_lines.setdefault(s.turn_idx, []).append(s.text)
            turn_roles[s.turn_idx] = s.role
        turn_tokens = {
            t: count_tokens(f"{turn_roles[t].upper()}: {' '.join(parts)}")
            for t, parts in turn_lines.items()
        }
        total = current
        added: list[_Sentence] = []

        for attempts, cand in enumerate(pool):
            if attempts >= 1200:  # safety valve; never unbounded work
                break
            if budget - total < cheapest:
                break
            parts = turn_lines.get(cand.turn_idx)
            if parts is None:
                new_tokens = count_tokens(f"{cand.role.upper()}: {cand.text}")
                delta = new_tokens + 1  # + newline separator
            else:
                new_tokens = count_tokens(f"{turn_roles[cand.turn_idx].upper()}: {' '.join([*parts, cand.text])}")
                delta = new_tokens - turn_tokens[cand.turn_idx]
            if total + delta > budget:
                continue
            turn_lines.setdefault(cand.turn_idx, []).append(cand.text)
            turn_roles[cand.turn_idx] = cand.role
            turn_tokens[cand.turn_idx] = new_tokens
            total += delta
            added.append(cand)
            # Periodic exact re-sync against the real renderer (bounded work).
            if len(added) % 64 == 0:
                total = count_tokens(self._render(selected + added))

        if not added:
            return text, len(selected)

        # Exact verification: drop from the tail until the real render fits. The
        # incremental accounting can be a few tokens optimistic at turn
        # boundaries, so this loop is what actually guarantees the budget.
        trial = self._render(selected + added)
        while added and count_tokens(trial) > budget:
            added.pop()
            trial = self._render(selected + added)
        if not added:
            return text, len(selected)
        selected.extend(added)
        return trial, len(selected)


# ---------------------------------------------------------------------------
# Optional LLMLingua-2 adapter (opt-in, never downloads anything)
# ---------------------------------------------------------------------------
class LLMLinguaExtractiveCompressor(Compressor):
    """Token-level extractive compression via an already-local LLMLingua-2 model.

    Off by default because the audit must be reproducible offline and because
    LLMLingua's default constructor downloads weights from HuggingFace. Enable
    only if you have the package *and* a local model directory::

        MEMFAITH_USE_LLMLINGUA=1
        MEMFAITH_LLMLINGUA_MODEL=/path/to/llmlingua-2-bert-base-multilingual-cased

    Any failure (import, load, compress) is logged and degrades to
    :class:`ExtractiveCompressor`, so a broken optional dependency can never
    change the experiment's shape.
    """

    name = "extractive_llmlingua"

    def __init__(self, target_ratio: float | None = None, model_path: str | None = None):
        self.target_ratio = DEFAULT_TARGET_RATIO if target_ratio is None else float(target_ratio)
        self.model_path = model_path or os.environ.get("MEMFAITH_LLMLINGUA_MODEL", "")
        self._fallback = ExtractiveCompressor(target_ratio=self.target_ratio)
        self._compressor = None
        self._load_error = ""

    def _load(self):
        if self._compressor is not None or self._load_error:
            return self._compressor
        if os.environ.get("MEMFAITH_USE_LLMLINGUA") != "1":
            self._load_error = "MEMFAITH_USE_LLMLINGUA!=1"
            return None
        if not self.model_path:
            self._load_error = "MEMFAITH_LLMLINGUA_MODEL not set (refusing to download weights)"
            return None
        try:  # pragma: no cover - optional dependency, absent in CI
            from llmlingua import PromptCompressor  # type: ignore

            self._compressor = PromptCompressor(
                model_name=self.model_path, use_llmlingua2=True, device_map="cpu"
            )
        except Exception as e:  # noqa: BLE001
            self._load_error = f"{type(e).__name__}: {str(e)[:160]}"
            self._compressor = None
        return self._compressor

    def compress(self, session: Session, client=None) -> CompressionResult:
        original_tokens = original_token_count(session)
        budget = target_budget(original_tokens, self.target_ratio, floor=160)
        transcript = render_transcript(session)
        comp = self._load()
        text = ""
        used_llmlingua = False
        if comp is not None:
            try:  # pragma: no cover - optional dependency
                result = comp.compress_prompt(
                    transcript,
                    ratio=self.target_ratio,
                    force_tokens=["\n", ":", "/", "."],
                )
                text = str(result.get("compressed_prompt", "")).strip()
                used_llmlingua = bool(text)
            except Exception as e:  # noqa: BLE001
                log.warning("llmlingua failed, falling back to sentence extractor: %s", e)

        if not text:
            out = self._fallback.compress(session, client)
            out.meta["llmlingua"] = False
            out.meta["llmlingua_error"] = self._load_error or "empty output"
            return out

        text, truncated = truncate_to_token_budget(text, budget, marker=" …")
        compressed_tokens = count_tokens(text)
        return CompressionResult(
            session_id=session.session_id,
            strategy=self.name,
            text=text,
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            ratio=(compressed_tokens / original_tokens) if original_tokens else 0.0,
            cost_tokens=0,
            meta={
                "method": "llmlingua2",
                "kept_sentences": text.count("\n") + 1 if text else 0,
                "total_sentences": len(split_sentences(transcript)),
                "budget_tokens": budget,
                "target_ratio": self.target_ratio,
                "llmlingua": used_llmlingua,
                "model_path": self.model_path,
                "truncated": truncated,
            },
        )


__all__ = [
    "DEFAULT_TARGET_RATIO",
    "ExtractiveCompressor",
    "LLMLinguaExtractiveCompressor",
    "METHOD",
    "split_sentences",
]
