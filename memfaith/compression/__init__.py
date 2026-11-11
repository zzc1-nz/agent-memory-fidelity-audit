"""Compression strategies under audit.

Two treatment arms, chosen to isolate *what kind* of compression loses fidelity:

``summary_hierarchical``
    LLM map-reduce abstraction (OpenMemX-flavoured "hierarchical abstraction +
    forgetting"). Rewrites text, so it can drop specifics but should keep gist.

``extractive``
    Deterministic sentence selection (AgentMem/LLMLingua-flavoured "verbatim"
    compression). Keeps exact strings, but can drop the context around them and
    may keep a stale value while dropping the sentence that superseded it.

``get_compressor(name)`` is the only entry point the runner needs. It takes a
single argument, reads its budgets from :mod:`memfaith.config` defaults, and
raises ``KeyError`` for unknown strategies so a typo fails loudly instead of
silently producing a third arm.
"""

from __future__ import annotations

from memfaith.compression.base import Compressor
from memfaith.compression.extractive import (
    LLMLinguaExtractiveCompressor,
    ExtractiveCompressor,
)
from memfaith.compression.summary import HierarchicalSummaryCompressor

# Registry of the arms the audit reports on. The optional LLMLingua adapter is
# deliberately NOT in here: it is an ablation, not a headline strategy, and it
# must never be picked by default (it would change what the audit measures).
STRATEGIES: dict[str, type[Compressor]] = {
    HierarchicalSummaryCompressor.name: HierarchicalSummaryCompressor,
    ExtractiveCompressor.name: ExtractiveCompressor,
}

# Extra names resolvable on request but excluded from the default matrix.
OPTIONAL_STRATEGIES: dict[str, type[Compressor]] = {
    LLMLinguaExtractiveCompressor.name: LLMLinguaExtractiveCompressor,
}


def get_compressor(name: str, **kwargs) -> Compressor:
    """Instantiate the compression strategy called ``name``.

    Parameters
    ----------
    name:
        One of ``"summary_hierarchical"``, ``"extractive"`` (or
        ``"extractive_llmlingua"`` for the opt-in ablation).
    **kwargs:
        Optional overrides forwarded to the constructor (e.g.
        ``target_ratio=0.15``). The runner passes nothing, so budgets come from
        the ``RunConfig`` defaults.

    Raises
    ------
    KeyError
        If ``name`` is not a known strategy. The message lists the valid names.
    """
    registry = {**STRATEGIES, **OPTIONAL_STRATEGIES}
    try:
        cls = registry[name]
    except KeyError:
        valid = ", ".join(sorted(registry))
        raise KeyError(f"unknown compression strategy {name!r}; expected one of: {valid}") from None
    return cls(**kwargs)


__all__ = [
    "Compressor",
    "ExtractiveCompressor",
    "HierarchicalSummaryCompressor",
    "LLMLinguaExtractiveCompressor",
    "OPTIONAL_STRATEGIES",
    "STRATEGIES",
    "get_compressor",
]
