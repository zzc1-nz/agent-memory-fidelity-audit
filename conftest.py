"""Project-level pytest configuration.

`memfaith` is importable from the repo root. pytest itself is not installed into the
global interpreter (pip cannot unpack wheels in this sandbox), so it is borrowed from
the workspace's shared `.pylibs` directory.

Sandbox note: pytest's built-in `tmpdir` plugin creates its base directory with
`mkdtemp`, which produces a directory this sandbox's ACLs let us create but not
re-scan, so its teardown hook raises `PermissionError` on an otherwise green run.
The plugin is therefore disabled (`-p no:tmpdir`, set in scripts/test.py) and a plain
`tmp_path` fixture is provided below, backed by a normal project-local directory.
"""

import os
import sys
import uuid

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Make a workspace-shared pytest visible if the interpreter lacks it.
_SHARED = os.path.abspath(os.path.join(_HERE, "..", ".pylibs"))
if os.path.isdir(_SHARED) and _SHARED not in sys.path:
    sys.path.append(_SHARED)


@pytest.fixture
def tmp_path():
    """A disposable directory under the project, replacing pytest's built-in one."""
    root = os.path.join(_HERE, ".pytmp", f"t-{uuid.uuid4().hex[:12]}")
    os.makedirs(root, exist_ok=True)
    import pathlib

    yield pathlib.Path(root)
    # Best-effort cleanup; never fail a passing test because of sandbox ACLs.
    import shutil

    shutil.rmtree(root, ignore_errors=True)
