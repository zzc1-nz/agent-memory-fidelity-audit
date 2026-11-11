"""Shared data contracts for the memory-fidelity audit.

Every module (dataset builder, compressors, probe generator, evaluator, report)
agrees on these structures. They are plain dicts at rest (JSONL/JSON on disk) with
light dataclass wrappers for type safety in code.

Core objects
------------
Session      : one synthetic long interaction history (20-30 turns) + its ground truth.
Turn         : a single message in the history.
Fact         : a ground-truth information unit, tagged with the attributes the
               audit needs to explain compression failures (frequency, salience,
               embedding depth, update history).
Probe        : a question of one of three kinds (high-level / specific / multi-hop)
               with a reference answer and the fact ids it depends on.
Compression  : the output of one strategy on one session, with token accounting.
Answer       : one (probe, condition) model response, plus its judged verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Taxonomy constants (kept as Literals so typos fail loudly)
# ---------------------------------------------------------------------------
TaskDomain = Literal["coding", "research", "daily"]
TurnRole = Literal["user", "assistant", "tool"]

ProbeKind = Literal["high_level", "specific_fact", "multi_hop"]
StrategyName = Literal["full_context", "summary_hierarchical", "extractive"]
Condition = Literal["full", "compressed"]

# Attributes used for the failure analysis.
Salience = Literal["foreground", "background"]
Embedding = Literal["standalone", "embedded", "deep_buried"]
Stability = Literal["stable", "superseded"]


@dataclass
class Turn:
    idx: int
    role: TurnRole
    text: str
    # provenance for auditing: which fact ids this turn mentions
    fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fact:
    """A ground-truth information unit.

    The audit's central question is which *facts* survive compression. Each fact
    carries the covariates we need for the failure analysis:

      frequency      : how many turns mention it (high -> redundant -> safe to drop)
      salience       : foreground (user emphasised it) vs background (mentioned in passing)
      embedding      : standalone bullet vs embedded in a long narrative vs deep-buried
                       inside an assistant answer with lots of surrounding text
      stability      : stable vs superseded (a later turn overwrote the value)
      critical       : whether a probe depends on it
      turn_idx       : first mention
    """

    fact_id: str
    kind: str                      # path / package / api_key_location / citation / date / preference / ...
    value: str                     # canonical value the probe expects
    domain: TaskDomain
    first_turn: int
    mentions: list[int] = field(default_factory=list)   # turn indices mentioning it
    salience: Salience = "background"
    embedding: Embedding = "embedded"
    stability: Stability = "stable"
    critical: bool = True
    superseded_by: str | None = None
    surface_forms: list[str] = field(default_factory=list)
    # For superseded facts: the value that a later turn overwrote. Kept so the
    # failure analysis can detect the worst outcome — a compressor that retains
    # the STALE value and drops the update, producing a confidently wrong answer.
    superseded_old_value: str | None = None

    @property
    def frequency(self) -> int:
        return len(self.mentions)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frequency"] = self.frequency
        return d


@dataclass
class Session:
    session_id: str
    domain: TaskDomain
    turns: list[Turn]
    facts: list[Fact]
    seed: int = 0
    title: str = ""
    full_tokens: int = 0

    def fact_index(self) -> dict[str, Fact]:
        return {f.fact_id: f for f in self.facts}

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "seed": self.seed,
            "title": self.title,
            "full_tokens": self.full_tokens,
            "turns": [t.to_dict() for t in self.turns],
            "facts": [f.to_dict() for f in self.facts],
        }


@dataclass
class Probe:
    probe_id: str
    session_id: str
    kind: ProbeKind
    question: str
    reference_answer: str
    fact_ids: list[str]
    domain: TaskDomain
    # optional: distractors / must-not-say content for multi-hop
    hops: int = 1
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompressionResult:
    session_id: str
    strategy: StrategyName
    text: str
    compressed_tokens: int
    original_tokens: int
    ratio: float                    # compressed / original
    cost_tokens: int                # tokens *spent to produce* the compression
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    probe_id: str
    session_id: str
    strategy: StrategyName
    condition: Condition
    answer_text: str
    correct: bool | None = None
    judge_raw: str = ""
    judge_confidence: float | None = None
    context_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(path: str, rows: list[dict]) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    import json

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
