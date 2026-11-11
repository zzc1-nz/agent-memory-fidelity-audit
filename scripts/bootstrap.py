"""Vendor the tiktoken wheel into ./vendor/tiktoken (no global install needed).

pip cannot unpack wheels inside the DSH sandbox temp directory, so we download the
wheel straight from PyPI's JSON API and unzip it ourselves.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "vendor", "tiktoken")


def main() -> int:
    os.makedirs(TARGET, exist_ok=True)
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    meta = json.loads(urllib.request.urlopen("https://pypi.org/pypi/tiktoken/json", timeout=60).read())
    url = None
    for u in meta["urls"]:
        fn = u["filename"]
        if py in fn and "win_amd64" in fn and f"{py}t" not in fn:
            url = u["url"]
            print("wheel:", fn)
            break
    if url is None:
        print(f"no {py} win_amd64 wheel published; trying pure-python fallback")
        return 1

    blob = urllib.request.urlopen(url, timeout=300).read()
    whl = os.path.join(TARGET, "_tiktoken.whl")
    with open(whl, "wb") as f:
        f.write(blob)
    with zipfile.ZipFile(whl) as z:
        z.extractall(TARGET)
    os.remove(whl)

    sys.path.insert(0, TARGET)
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    print("tiktoken", tiktoken.__version__, "| sample:", len(enc.encode("hello 你好 world")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
