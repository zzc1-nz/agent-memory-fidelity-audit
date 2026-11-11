"""Vendored tiktoken loader.

tiktoken was installed from a wheel into ./vendor/tiktoken so the project does not
depend on a global site-packages install (pip cannot write wheels into the sandbox
temp dir). This module makes the vendored copy importable and exposes a single
`count_tokens` used by every module, so token accounting is consistent across the
dataset builder, the compressors and the metrics layer.
"""

from __future__ import annotations

import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "tiktoken")

if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

try:
    import tiktoken
except Exception as exc:  # pragma: no cover - fallback path
    tiktoken = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        if tiktoken is None:
            raise RuntimeError(
                "tiktoken is unavailable; run `python scripts/bootstrap.py` to vendor it. "
                f"Original import error: {_IMPORT_ERROR}"
            )
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens with cl100k_base (the tokenizer family used by GPT-4/DeepSeek-era models)."""
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))


def count_message_tokens(messages: list[dict], overhead_per_message: int = 4) -> int:
    """Approximate chat-format token cost (content + per-message role/framing overhead)."""
    total = 0
    for m in messages:
        total += count_tokens(str(m.get("role", "")))
        total += count_tokens(str(m.get("content", "")))
        total += overhead_per_message
    return total


def encoder_name() -> str:
    return "cl100k_base"
