"""Shared machinery for the two compression strategies.

This module holds the parts that must behave *identically* across arms so the
audit is a fair comparison:

``Compressor``
    The structural contract from ``CONTRACT.md``. The eval harness only ever
    touches ``.name`` and ``.compress(session, client)``.

``ledger_delta`` / ``measure_ledger``
    Exact token accounting for the *cost* side of the headline metric. The
    audit's question is "what does one correct answer cost?", so compression
    spend has to be attributed to the compression call that caused it. The
    ``UsageLedger`` is append-only, so the reliable delta is "records appended
    during this call, summed over prompt+completion". We snapshot the record
    count before and after; no estimates, no model-reported context sizes.

    Caching caveat (important for reading results): ``LLMClient`` caches
    responses to disk and, on a cache hit, appends a ledger record with zero
    tokens because no tokens were actually spent. A warm re-run therefore
    reports ``cost_tokens == 0``. The *cold-cache* number is the authoritative
    one for cost-per-answer; ``meta["cache_hits"]`` is recorded so a reader can
    tell the two runs apart.

``truncate_to_token_budget`` / ``split_by_token_budget`` / ``fit_units_to_budget``
    Budget enforcement. The two arms are budget-aware (target ratio ~0.25) and
    a summary model cannot be trusted to respect a token count, so the final
    output is hard-fitted with a character binary search over the *same*
    tokenizer used everywhere else (``cl100k_base``). This keeps
    ``CompressionResult.compressed_tokens`` exact rather than approximate.

    ``fit_units_to_budget`` is the bullet-aware variant: it drops whole lines
    rather than cutting one in half, because a truncated path
    (".../srv/app/secr") is not a fact, it is noise that still costs tokens. It
    can also take a ``priority`` ordering so the cut removes the *least
    informative* bullet instead of always the last one — important here, because
    the end of a session is where superseded facts hold their current value, and
    a positional cut would manufacture an "update-forgetting" artifact that
    belongs to the compression paradigm, not to our truncation step.

``concrete_value_score``
    The shared definition of "this text carries an auditable fact" — paths,
    version pins, hosts, ids, dates, secrets, quoted literals, named choices
    ("we standardised on poetry") and explicit remember/important cues. Both arms
    use it, for different purposes: the extractive arm scores candidates with it,
    and the summary arm uses it to decide *which* bullets to drop when the model
    overshoots the budget. Sharing one definition keeps the comparison honest:
    neither arm is given a better notion of "valuable" than the other.

``stale_and_latest_positions``
    Supersession detection, also shared. A slot is an updatable concrete value
    with its digits masked (``v2.1.0`` and ``v2.2.0`` are the same slot); when two
    *different* values occupy one slot the earlier is stale and the later is
    current. Both arms demote the stale mention, because probes ask for the
    current value — keeping the old one is a failure of the paradigm under test,
    not a bonus point for our selector.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from memfaith.schema import CompressionResult, Session
from memfaith.tokens import count_tokens

T = TypeVar("T")

# Minimum character-level prefix kept when truncating. Guards the pathological
# case where a single character costs more than the whole budget.
_MIN_KEEP_CHARS = 1


# ---------------------------------------------------------------------------
# Compressor protocol
# ---------------------------------------------------------------------------
class Compressor(Protocol):
    """A memory-compression strategy.

    Implementations must be pure with respect to the *session*: the same
    ``(session, client-responses)`` pair must produce the same text, so a
    re-run of the audit reproduces byte-identical ``compressions.jsonl``.

    ``compress`` must never raise for ordinary model flakiness; the arms
    degrade to a documented fallback instead (see ``meta["fallbacks"]``).
    """

    name: str

    def compress(self, session: Session, client: Any) -> CompressionResult: ...


# ---------------------------------------------------------------------------
# Ledger accounting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerDelta:
    """Tokens/calls appended to an ``LLMClient`` ledger during one operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0

    @property
    def cost_tokens(self) -> int:
        """Prompt + completion tokens actually spent (cached calls cost nothing)."""
        return self.prompt_tokens + self.completion_tokens

    def to_meta(self) -> dict[str, int]:
        """Compact form for ``CompressionResult.meta``."""
        return {
            "cost_tokens": self.cost_tokens,
            "cost_prompt_tokens": self.prompt_tokens,
            "cost_completion_tokens": self.completion_tokens,
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "llm_failures": self.failures,
        }


def _ledger_snapshot(client: Any) -> tuple[str, int] | None:
    """Return a comparable position in the client's ledger, or ``None``.

    Two shapes are supported so this helper also works with a minimal fake:

    * ``ledger.records`` — a list of dicts (the real ``Ledger``); position is
      the record count, and the delta sums the appended records.
    * ``ledger.totals()["grand_total"]["total_tokens"]`` — for clients whose
      ledger only exposes aggregates; the delta is the difference.
    """
    ledger = getattr(client, "ledger", None)
    if ledger is None:
        return None
    records = getattr(ledger, "records", None)
    if records is not None:
        return ("records", len(records))
    totals = getattr(ledger, "totals", None)
    if callable(totals):
        try:
            return ("totals", int(totals()["grand_total"]["total_tokens"]))
        except Exception:  # noqa: BLE001 - accounting must never break the run
            return None
    return None


def _ledger_delta_since(client: Any, snapshot: tuple[str, int] | None) -> LedgerDelta:
    if snapshot is None:
        return LedgerDelta()
    kind, position = snapshot
    ledger = getattr(client, "ledger", None)
    if kind == "records":
        records = list(getattr(ledger, "records", [])[position:])
        return LedgerDelta(
            prompt_tokens=sum(int(r.get("prompt_tokens", 0) or 0) for r in records),
            completion_tokens=sum(int(r.get("completion_tokens", 0) or 0) for r in records),
            calls=len(records),
            cache_hits=sum(1 for r in records if r.get("cached")),
            failures=sum(1 for r in records if not r.get("ok", True)),
        )
    total = int(ledger.totals()["grand_total"]["total_tokens"])
    return LedgerDelta(completion_tokens=max(0, total - position))


@contextlib.contextmanager
def measure_ledger(client: Any) -> Iterator[Callable[[], LedgerDelta]]:
    """Context manager yielding a zero-arg getter for the ledger delta so far.

    Usage::

        with measure_ledger(client) as delta:
            text = client.chat("summarizer", messages, max_tokens=1100)
            spent = delta().cost_tokens

    The delta is computed from *appended records*, so retries and cache hits
    are accounted for the same way the client itself accounts for them.
    """
    snapshot = _ledger_snapshot(client)
    yield lambda: _ledger_delta_since(client, snapshot)


def ledger_delta(client: Any, fn: Callable[[], T]) -> tuple[T, LedgerDelta]:
    """Run ``fn`` and return ``(result, tokens spent inside it)``.

    This is the function form of :func:`measure_ledger`; either is fine, the
    function form is handy when the work is a single expression.
    """
    snapshot = _ledger_snapshot(client)
    result = fn()
    return result, _ledger_delta_since(client, snapshot)


# ---------------------------------------------------------------------------
# Shared "is this a concrete, auditable fact?" model
# ---------------------------------------------------------------------------
# (name, compiled regex, weight). Weights are tuned so one concrete value
# outweighs a long stretch of conversational prose. These are exactly the fact
# kinds the dataset injects and the probes ask about, so a high score here is a
# proxy for "a probe could depend on this sentence".
CONCRETE_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    # File paths. Deliberately permissive about the root (any absolute path with
    # at least two segments, plus home-relative `~/...` and relative `./...`)
    # because real sessions use roots we did not anticipate (/run/secrets,
    # ~/.config, ./infra) and a missed path is a missed probe answer. The
    # two-segment requirement keeps prose like "and/or" from matching.
    ("path", re.compile(
        r"(?<![\w/])/(?:[\w.-]+/)+[\w.-]+"          # /run/secrets/svc_api_key
        r"|(?<![\w])~/(?:[\w.-]+/)*[\w.-]+"          # ~/.config/svc/token.env
        r"|(?<![\w])\.{1,2}/(?:[\w.-]+/)*[\w.-]+"    # ./infra/prod.tf
        r"|[A-Za-z]:\\[^\s,;\"']+"                   # C:\Users\...\prod.env
        r"|\b[\w.-]+\.(?:py|js|ts|tsx|jsx|json|yaml|yml|toml|env|md|csv|parquet"
        r"|sql|sh|cfg|ini|tf|key|pem|crt|lock)\b"    # bare filenames with a known suffix
    ), 3.2),
    ("url", re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE), 3.0),
    # Version pins and constraints: `pkg==1.2.3`, `python>=3.12`, `~=2.0`, `v1.2.3`.
    # The comparison operators matter — a bare name is unidentifiable, but
    # "python>=3.12" is a constraint a probe can ask for verbatim.
    ("version_pin", re.compile(
        r"\b[\w.-]+\s*(?:==|>=|<=|~=|!=)\s*\d[\w.]*|\bv?\d+\.\d+\.\d+(?:-[\w.]+)?\b"
    ), 3.2),
    ("dotted_id", re.compile(r"\b[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*){2,}\b"), 2.2),
    ("arxiv", re.compile(r"\b(?:arXiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE), 3.6),
    ("doi", re.compile(r"\b10\.\d{4,9}/\S+"), 3.4),
    ("date", re.compile(
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b",
        re.IGNORECASE,
    ), 2.8),
    ("deadline", re.compile(
        r"\bdeadline\b|\bby (?:eod|eow|friday|monday|tomorrow|next week)\b|\bEOD\b|\bEOW\b",
        re.IGNORECASE,
    ), 2.4),
    ("port", re.compile(r":\d{2,5}\b"), 2.4),
    ("host", re.compile(
        r"\b(?:[\w-]+\.)+(?:com|ai|io|dev|org|net|internal|local|corp|cloud)\b"
    ), 2.6),
    ("secret", re.compile(
        r"\bapi[_ ]?key\b|\bsecret\b|\bcredential\b|\btoken\b|\bkeychain\b|\bpassword\b",
        re.IGNORECASE,
    ), 2.8),
    ("quoted", re.compile(r"\"[^\"]{2,}\"|`[^`]{2,}`|'[^']{3,}'"), 1.4),
    ("allcaps", re.compile(r"\b[A-Z]{3,}\b"), 1.2),
    ("cue", re.compile(
        r"\bremember\b|\bimportant\b|\bnote that\b|\bnever\b|\balways\b|\bmust\b"
        r"|\bprefer\b|\bconstraint\b|\bdo not\b|\bdon't\b",
        re.IGNORECASE,
    ), 2.0),
    ("decision", re.compile(
        r"\bdecided\b|\bwe'll use\b|\bwe will use\b|\bswitched to\b|\bmoved to\b"
        r"|\binstead of\b|\bchose\b|\bsettled on\b",
        re.IGNORECASE,
    ), 1.8),
    # Bare identifiers have no syntax to key on, so they are caught by the
    # *sentence frame* that names them: "we standardised on poetry", "the package
    # manager is uv", "we use pydantic". Without this, a fact whose value is just
    # a name ("poetry", "uv") is invisible to a purely lexical scorer — and
    # `package` is one of the fact kinds the probes ask about.
    ("named_choice", re.compile(
        r"\b(?:standardised|standardized|settled|decided)\s+on\s+([A-Za-z][\w.+-]*)"
        r"|\b(?:package manager|build tool|runtime|framework|library|orm|test runner)\s+"
        r"(?:is|will be|should be)\s+([A-Za-z][\w.+-]*)"
        r"|\b(?:we|i)\s+(?:will\s+)?(?:use|prefer|switched to|moved to|chose)\s+([A-Za-z][\w.+-]*)",
        re.IGNORECASE,
    ), 2.6),
    ("number_unit", re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|minutes|hours|days|kb|mb|gb|tb|%|percent)\b",
        re.IGNORECASE,
    ), 1.0),
)


def concrete_value_score(text: str) -> float:
    """Additive score for how many auditable concrete values ``text`` contains.

    Deliberately simple and additive: a sentence with a path *and* a version pin
    scores higher than one with only a path, and the per-kind weight says which
    kinds the audit cares about most. Repeated hits of one kind are discounted
    (``1 + 0.25*(hits-1)``) so a wall of URLs does not outrank a diverse,
    information-dense bullet.
    """
    total = 0.0
    for _name, rx, weight in CONCRETE_VALUE_PATTERNS:
        hits = len(rx.findall(text))
        if hits:
            total += weight * (1.0 + 0.25 * (hits - 1))
    return total


# Concrete values that can be *updated* over a session: paths, version pins,
# hosts, quoted literals, paper ids. Used for the "latest value wins" pass.
CONCRETE_ENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z]:\\[^\s,;\"']+"),
    re.compile(r"(?<![\w/])/(?:[\w.-]+/)+[\w.-]+"),
    re.compile(r"(?<![\w])~/(?:[\w.-]+/)*[\w.-]+"),
    re.compile(r"\b[\w.-]+\s*(?:==|>=|<=|~=|!=)\s*\d[\w.]*"),
    re.compile(r"\bv?\d+\.\d+\.\d+(?:-[\w.]+)?\b"),
    re.compile(r"\b(?:[\w-]+\.)+(?:com|ai|io|dev|org|net|internal|local|corp|cloud)\b"),
    re.compile(r"\"[^\"]{2,}\"|`[^`]{2,}`"),
    re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"),
)


def concrete_entities(text: str) -> frozenset[str]:
    """Extract comparable concrete values (lowercased) from ``text``."""
    found: set[str] = set()
    for rx in CONCRETE_ENTITY_PATTERNS:
        for m in rx.finditer(text):
            value = m.group(0).strip().strip("\"'`")
            if len(value) >= 3:
                found.add(value.lower())
    return frozenset(found)


def entity_family(entity: str) -> str:
    """Map a concrete value to its *slot*, masking the part that gets updated.

    ``v2.1.0`` and ``v2.2.0`` are different strings but the same slot: one
    superseded the other. Masking digits lets the scorer notice that a value
    *changed*, which is exactly what the audit's "update/forgetting" probes ask
    about. Two mentions of the *identical* string are redundancy (handled by
    dedup), not an update, so they are deliberately not treated as supersession.
    """
    return re.sub(r"\d+", "#", entity)


def stale_and_latest_positions(
    entity_sets: list[frozenset[str]],
) -> tuple[set[int], set[int]]:
    """Indices holding a superseded value, and indices holding the newest one.

    Takes precomputed entity sets so callers that already extracted entities do
    not pay for it twice. A slot (``entity_family``) counts as superseded only
    when at least two *different* values occupy it; repeating the same value is
    redundancy, not an update. Both arms use this so neither keeps a value the
    probes will mark wrong — the audit is about what compression loses, not about
    whether our selector can read a correction.
    """
    slot_hits: dict[str, list[tuple[int, str]]] = {}
    for i, entities in enumerate(entity_sets):
        for entity in entities:
            slot_hits.setdefault(entity_family(entity), []).append((i, entity))

    stale: set[int] = set()
    latest: set[int] = set()
    for hits in slot_hits.values():
        if len({value for _i, value in hits}) < 2:
            continue
        last = max(i for i, _value in hits)
        latest.add(last)
        stale.update(i for i, _value in hits if i < last)
    return stale, latest


def stale_positions(texts: list[str]) -> set[int]:
    """Indices of ``texts`` holding a value that a LATER text replaced."""
    return stale_and_latest_positions([concrete_entities(t) for t in texts])[0]


def latest_positions(texts: list[str]) -> set[int]:
    """Indices of ``texts`` holding the newest value of an updated slot."""
    return stale_and_latest_positions([concrete_entities(t) for t in texts])[1]


# ---------------------------------------------------------------------------
# Rendering (single source of truth for the "full context" text)
# ---------------------------------------------------------------------------
def fallback_render_transcript(session: Session) -> str:
    """Render ``session`` exactly like ``memfaith.dataset.generator`` does.

    The dataset module owns the canonical renderer; this copy exists so the
    compressors stay importable/testable while that module is still being
    written, and as a defensive fallback if it ever moves. Format is the one
    fixed in ``CONTRACT.md``: ``"USER: ...\\nASSISTANT: ...\\n"``.
    """
    return "".join(f"{t.role.upper()}: {t.text}\n" for t in session.turns)


def render_transcript(session: Session) -> str:
    """Delegate to the dataset renderer when available, else use the fallback."""
    try:  # pragma: no cover - depends on import order across modules
        from memfaith.dataset.generator import render_transcript as _canonical

        return _canonical(session)
    except Exception:  # noqa: BLE001 - dataset module may not exist yet
        return fallback_render_transcript(session)


def original_token_count(session: Session) -> int:
    """``session.full_tokens`` when the builder filled it, else recompute it.

    ``CONTRACT.md`` defines ``original_tokens = session.full_tokens``; we only
    recompute for hand-built fixtures (e.g. unit tests) where it is still 0.
    """
    if session.full_tokens > 0:
        return session.full_tokens
    return count_tokens(render_transcript(session))


# ---------------------------------------------------------------------------
# Exact budget enforcement
# ---------------------------------------------------------------------------
def _largest_prefix_within(text: str, budget: int) -> int:
    """Length of the longest prefix of ``text`` whose token count fits.

    Binary search over characters, then a short linear safety pass. Prefix
    token counts are *almost* monotone in the prefix length, but BPE is greedy
    on the whole string, so we verify the candidate instead of trusting the
    invariant.
    """
    if budget <= 0:
        return 0
    if count_tokens(text) <= budget:
        return len(text)

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    # Safety pass: never return a prefix that is over budget.
    while lo > 0 and count_tokens(text[:lo]) > budget:
        lo -= 1
    return lo


def _snap_to_boundary(text: str, cut: int, *, min_fraction: float = 0.6) -> int:
    """Pull ``cut`` back to the nearest newline (then space) if not too costly."""
    if cut >= len(text) or cut <= 0:
        return cut
    floor = int(cut * min_fraction)
    newline = text.rfind("\n", 0, cut)
    if newline >= floor:
        return newline + 1
    space = text.rfind(" ", 0, cut)
    if space >= floor:
        return space + 1
    return cut


def truncate_to_token_budget(
    text: str,
    budget: int,
    *,
    marker: str = "",
    prefer_boundary: bool = True,
) -> tuple[str, bool]:
    """Fit ``text`` into ``budget`` tokens. Returns ``(fitted_text, truncated)``.

    Guarantee: ``count_tokens(fitted_text) <= max(0, budget)``. This is what
    makes ``CompressionResult.compressed_tokens`` an exact number rather than a
    hope about what the summarizer did.

    ``marker`` (e.g. ``" …"``) is appended only if it still fits, so the
    guarantee holds even when the budget is tiny.
    """
    budget = max(0, int(budget))
    if not text:
        return "", False
    if count_tokens(text) <= budget:
        return text, False

    cut = _largest_prefix_within(text, budget)
    if prefer_boundary:
        cut = _snap_to_boundary(text, cut)
    cut = max(cut, min(_MIN_KEEP_CHARS, len(text)))
    out = text[:cut].rstrip()

    if marker and count_tokens(out + marker) <= budget:
        out += marker

    # Belt and braces: the guarantee must hold exactly.
    while out and count_tokens(out) > budget:
        out = out[:-1]
    if not out:
        # Budget smaller than one character: return the empty string, which
        # always satisfies count_tokens("") == 0 <= budget.
        return "", True
    return out, True


def fit_units_to_budget(
    units: list[str],
    budget: int,
    *,
    separator: str = "\n",
    marker: str = "",
    priority: list[int] | None = None,
) -> tuple[str, bool, int]:
    """Keep whole ``units`` (bullets/lines) up to ``budget`` tokens.

    Truncating mid-bullet leaves a dangling half-fact ("- USER: The production
    API key lives at /srv/app/secr"), which is worse than dropping the bullet:
    a partial path is not a fact, it is noise that still costs tokens. So the
    summary arm prefers whole units and only falls back to character truncation
    when even the first unit does not fit.

    ``priority`` (optional) lists unit indices from most to least important. When
    given, units are admitted in that order and the survivors are re-emitted in
    their original order. This matters for the audit: plain prefix truncation
    always drops the *tail* of the session, and the tail is exactly where an
    updated ("superseded") value lives. Dropping by importance instead of by
    position keeps the truncation from silently manufacturing an
    "update-forgetting" artifact that belongs to the compression paradigm.

    Returns ``(text, truncated, units_kept)``.
    """
    budget = max(0, int(budget))
    order = list(range(len(units))) if priority is None else [i for i in priority if 0 <= i < len(units)]
    admitted: set[int] = set()
    for i in order:
        candidate = separator.join(units[j] for j in sorted([*admitted, i]))
        if count_tokens(candidate) > budget:
            continue
        admitted.add(i)

    if admitted:
        kept = sorted(admitted)
        text = separator.join(units[i] for i in kept)
        truncated = len(kept) < len(units)
        if truncated and marker:
            with_marker = text + marker
            if count_tokens(with_marker) <= budget:
                text = with_marker
        return text, truncated, len(kept)

    # Nothing fit whole: fall back to a hard character cut of the first unit(s).
    text, truncated = truncate_to_token_budget(separator.join(units), budget, marker=marker)
    return text, truncated, 1 if text else 0


def split_by_token_budget(text: str, budget: int) -> list[str]:
    """Hard-split ``text`` into consecutive pieces of at most ``budget`` tokens.

    Used only for a single turn that is larger than a whole chunk window; normal
    chunking keeps turn boundaries intact because losing "who said what" is a
    compression artifact we are auditing, not one we want to introduce.
    """
    budget = max(1, int(budget))
    pieces: list[str] = []
    rest = text
    while rest:
        if count_tokens(rest) <= budget:
            pieces.append(rest)
            break
        cut = _largest_prefix_within(rest, budget)
        cut = _snap_to_boundary(rest, cut)
        cut = max(cut, 1)
        pieces.append(rest[:cut])
        rest = rest[cut:]
        while rest and rest[0] == "\n":
            rest = rest[1:]
    return pieces


def target_budget(original_tokens: int, ratio: float, *, min_tokens: int = 0) -> int:
    """Token budget for a compressed memory: ``ratio * original_tokens``.

    The ratio is the contract, so the budget is derived from it directly and is
    never rounded *up* past it. ``min_tokens`` is an escape hatch for smoke
    tests on very short sessions (it can push the achieved ratio above the
    target, so the default is 0 and the audit does not use it).
    """
    if original_tokens <= 0:
        return 0
    budget = int(original_tokens * max(0.0, ratio))
    if min_tokens:
        budget = max(budget, int(min_tokens))
    return max(0, min(budget, original_tokens))


__all__ = [
    "CONCRETE_ENTITY_PATTERNS",
    "CONCRETE_VALUE_PATTERNS",
    "Compressor",
    "LedgerDelta",
    "concrete_entities",
    "concrete_value_score",
    "entity_family",
    "fallback_render_transcript",
    "fit_units_to_budget",
    "latest_positions",
    "ledger_delta",
    "measure_ledger",
    "original_token_count",
    "render_transcript",
    "split_by_token_budget",
    "stale_and_latest_positions",
    "stale_positions",
    "target_budget",
    "truncate_to_token_budget",
]
