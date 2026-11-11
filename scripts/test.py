"""One-command test runner that works without a global pytest install.

Usage:  python scripts/test.py [pytest args...]

pytest is not installable into the sandbox interpreter (pip cannot unpack wheels in
the temp dir), so this script locates a pytest already present in the workspace's
shared `.pylibs` and runs it with the repo root on sys.path.

`-p no:tmpdir` disables pytest's built-in temp-dir plugin: it creates its base
directory with `mkdtemp`, which this sandbox lets us create but not re-scan, so its
teardown raises PermissionError on otherwise-green runs. conftest.py supplies a
`tmp_path` fixture instead.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.abspath(os.path.join(ROOT, "..", ".pylibs"))

sys.path.insert(0, ROOT)
if os.path.isdir(SHARED):
    sys.path.append(SHARED)

try:
    import pytest
except ImportError:
    print("pytest not found. Install it or point PYTHONPATH at a directory containing pytest.")
    raise SystemExit(2)

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests", "-q"]
    args = ["-p", "no:tmpdir", "-p", "no:cacheprovider"] + args
    raise SystemExit(pytest.main(args))
