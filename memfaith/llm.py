"""LLM client: OpenAI-compatible, retrying, disk-cached, usage-accounted.

Why this exists
---------------
The audit's headline metric is *cost per correct answer*, so every token the
experiment spends must be attributed to a (role, session, strategy, question)
cell. This client is the single choke point for that accounting: every call goes
through `chat()` which records prompt/completion tokens, latency, retries and
failures into a shared `UsageLedger`.

Roles
-----
Three independent model identities are used so the judge never grades its own
generation (self-bias control):

  summarizer  -> writes hierarchical summaries
  answerer    -> answers probe questions (the "agent under audit")
  judge       -> grades answers against reference facts

Configure via env or constructor. Defaults target a keyless public gateway; any
OpenAI-compatible endpoint works, including a local one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Defaults: keyless public gateway that routes to free models.
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get("MEMFAITH_BASE_URL", "https://api.kilo.ai/api/openrouter/v1")
DEFAULT_API_KEY = os.environ.get("MEMFAITH_API_KEY", "")

DEFAULT_MODELS = {
    "summarizer": os.environ.get("MEMFAITH_MODEL_SUMMARIZER", "kilo-auto/free"),
    "answerer": os.environ.get("MEMFAITH_MODEL_ANSWERER", "nvidia/nemotron-3.5-super-vl:free"),
    "judge": os.environ.get("MEMFAITH_MODEL_JUDGE", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    "extractive": os.environ.get("MEMFAITH_MODEL_EXTRACTIVE", "kilo-auto/free"),
}


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------
@dataclass
class Ledger:
    """Thread-safe token/cost accounting keyed by (role, tag)."""

    records: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw: Any) -> None:
        with self._lock:
            self.records.append(kw)

    def totals(self) -> dict:
        with self._lock:
            recs = list(self.records)
        out: dict[str, dict] = {}
        for r in recs:
            key = f"{r['role']}"
            slot = out.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                        "total_tokens": 0, "failures": 0, "latency_s": 0.0})
            slot["calls"] += 1
            slot["prompt_tokens"] += r.get("prompt_tokens", 0)
            slot["completion_tokens"] += r.get("completion_tokens", 0)
            slot["total_tokens"] += r.get("total_tokens", 0)
            slot["latency_s"] = round(slot["latency_s"] + r.get("latency", 0.0), 2)
            if not r.get("ok"):
                slot["failures"] += 1
        grand = {
            "calls": sum(s["calls"] for s in out.values()),
            "prompt_tokens": sum(s["prompt_tokens"] for s in out.values()),
            "completion_tokens": sum(s["completion_tokens"] for s in out.values()),
            "total_tokens": sum(s["total_tokens"] for s in out.values()),
            "failures": sum(s["failures"] for s in out.values()),
        }
        return {"by_role": out, "grand_total": grand}

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"totals": self.totals(), "records": self.records}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
class DiskCache:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if enabled and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    @staticmethod
    def key(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        blob = json.dumps({"m": model, "msgs": messages, "t": temperature, "n": max_tokens},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, k: str):
        if not self.enabled:
            return None
        with self._lock:
            return self._data.get(k)

    def put(self, k: str, v: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[k] = v

    def flush(self) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with self._lock:
            data = dict(self._data)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        models: dict[str, str] | None = None,
        cache_path: str | None = None,
        ledger: Ledger | None = None,
        max_retries: int = 6,
        timeout: int = 180,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = dict(DEFAULT_MODELS)
        if models:
            self.models.update(models)
        self.ledger = ledger or Ledger()
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self.cache = DiskCache(cache_path, enabled=use_cache and bool(cache_path))
        self._flush_lock = threading.Lock()

    # -- low level ---------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "User-Agent": "memfaith-audit/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))

    def chat(
        self,
        role: str,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        tag: str = "",
        retries: int | None = None,
    ) -> str:
        """Return the assistant text. Retries transient errors; caches successes."""
        model = model or self.models.get(role)
        if not model:
            raise LLMError(f"no model configured for role {role!r}")

        ck = DiskCache.key(model, messages, temperature, max_tokens)
        hit = self.cache.get(ck)
        if hit is not None:
            self.ledger.add(role=role, tag=tag, model=model, ok=True, cached=True,
                            prompt_tokens=0, completion_tokens=0, total_tokens=0, latency=0.0)
            return hit

        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        attempts = self.max_retries if retries is None else retries
        last_err = ""
        for attempt in range(attempts):
            t0 = time.time()
            try:
                d = self._post(payload)
                msg = (d.get("choices") or [{}])[0].get("message") or {}
                content = msg.get("content")
                if content is None:
                    content = msg.get("reasoning_content") or ""
                content = (content or "").strip()
                usage = d.get("usage") or {}
                lat = round(time.time() - t0, 2)
                if not content:
                    # Some reasoning models burn the budget on hidden tokens.
                    raise LLMError("empty content")
                self.ledger.add(role=role, tag=tag, model=model, ok=True, cached=False,
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                                total_tokens=usage.get("total_tokens", 0), latency=lat)
                self.cache.put(ck, content)
                return content
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "ignore")[:200]
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {body}"
                # 401/403 are permanent (needs auth); 400 sometimes means model unavailable.
                fatal = e.code in (401, 403)
                if fatal:
                    break
            except Exception as e:  # noqa: BLE001 - network flakiness is expected
                last_err = f"{type(e).__name__}: {str(e)[:180]}"

            lat = round(time.time() - t0, 2)
            self.ledger.add(role=role, tag=tag, model=model, ok=False, cached=False,
                            prompt_tokens=0, completion_tokens=0, total_tokens=0, latency=lat,
                            error=last_err)
            if attempt < attempts - 1:
                backoff = min(2 ** attempt * 1.5, 30) + random.uniform(0, 1.0)
                if self.verbose:
                    print(f"    [retry {attempt + 1}/{attempts - 1}] {role}: {last_err[:90]} "
                          f"-> sleep {backoff:.1f}s")
                time.sleep(backoff)

        raise LLMError(f"role={role} model={model} failed after {attempts} attempts: {last_err}")

    # -- helpers -----------------------------------------------------------
    def flush(self) -> None:
        with self._flush_lock:
            self.cache.flush()

    def usage(self) -> dict:
        return self.ledger.totals()
