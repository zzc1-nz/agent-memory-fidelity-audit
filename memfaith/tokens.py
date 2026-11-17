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

# Vendored copy of tiktoken, produced by scripts/bootstrap.py. It is a *compiled*
# wheel for one specific Python version, so it must never shadow an interpreter that
# has its own working install: importing a cp314 build under cp310 raises, and worse,
# a silent failure here would make token accounting version-dependent. We only add the
# vendored path when the interpreter cannot already import tiktoken on its own, and we
# verify the vendored copy actually works before trusting it.
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "tiktoken")


def _try_import():
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")  # prove the native extension loads
        return tiktoken, None
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        return None, exc


tiktoken, _IMPORT_ERROR = _try_import()

if tiktoken is None and os.path.isdir(_VENDOR):
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    tiktoken, _IMPORT_ERROR = _try_import()

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
