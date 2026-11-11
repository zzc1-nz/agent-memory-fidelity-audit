"""Dataset package: synthetic long interaction histories with ground truth."""

from memfaith.dataset.generator import (
    build_sessions,
    load_sessions,
    render_transcript,
    save_sessions,
)

__all__ = ["build_sessions", "load_sessions", "render_transcript", "save_sessions"]
