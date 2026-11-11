"""Configuration: model roles, run sizes, paths.

The experiment's scale is driven from here so a smoke run and the full run differ
only by numbers, never by code paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.path.join(ROOT, ".cache")

# Free public gateway (no key needed). Override for a paid/local endpoint.
BASE_URL = os.environ.get("MEMFAITH_BASE_URL", "https://api.kilo.ai/api/openrouter/v1")
API_KEY = os.environ.get("MEMFAITH_API_KEY", "")

# Three distinct model identities => the judge never grades its own output.
# These are the OpenAI-compatible defaults; the Trae backend supplies its own
# equivalents in `memfaith.trae_client.ROLE_MODELS`.
MODELS = {
    "summarizer": os.environ.get("MEMFAITH_MODEL_SUMMARIZER", "kilo-auto/free"),
    "answerer": os.environ.get("MEMFAITH_MODEL_ANSWERER", "nvidia/nemotron-3.5-super-vl:free"),
    "judge": os.environ.get("MEMFAITH_MODEL_JUDGE", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    "extractive": os.environ.get("MEMFAITH_MODEL_EXTRACTIVE", "kilo-auto/free"),
}



@dataclass
class RunConfig:
    n_sessions: int = 30
    turns_min: int = 20
    turns_max: int = 30
    probes_per_session: int = 10
    # probe mix per session: high-level / specific / multi-hop
    probe_mix: tuple[int, int, int] = (3, 5, 2)
    strategies: tuple[str, ...] = ("summary_hierarchical", "extractive")
    seed: int = 20261110
    use_cache: bool = True
    # token budget target for the *compressed* context of the summary strategy
    summary_target_ratio: float = 0.25
    # extractive: keep this fraction of the original tokens
    extractive_target_ratio: float = 0.25
    concurrency: int = 4
    tag: str = "full"

    def probe_count(self) -> int:
        return self.probes_per_session


def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
