"""Trae gateway backend: the real LLM channel for this benchmark.

Why a bespoke client
--------------------
The audit needs three independent model identities (summarizer / answerer / judge)
and a few thousand calls. The only credential available in this environment is the
machine's own Trae login, which is *not* an OpenAI-compatible endpoint: it speaks
the SOLO `llm_utils_chat` SSE protocol with `Cloud-IDE-JWT` auth and per-install
identity headers. This module adapts that protocol to the same `chat()` surface
`memfaith.llm.LLMClient` exposes, so every metric and every token count works
unchanged.

Protocol notes (all verified against the live gateway)
------------------------------------------------------
* Auth is `Cloud-IDE-JWT <token>` plus `X-Ide-Token`, NOT `Bearer`.
* Chat lives on the region's *chat* gateway (`trae-api-cn.mchost.guru`), not on the
  `api.trae.cn` pay host and not on the `solo.trae.cn` remote-directory host.
* The body is `{messages, model, config_name, function, stream:true}`; the wire id
  is the model's `config_name`, and a model is only callable through a function
  whose roster lists it.
* The response is SSE: `event:output` frames carry `response` (visible text) and
  `reasoning_content` (hidden chain). We concatenate `response`; reasoning is
  counted separately so the token accounting stays honest.
* `event:error` frames carry a refusal code (4001 = model not served under this
  function, 4011 = rate limited on this route).

Token accounting
----------------
The gateway does not always return a usage frame, so prompt tokens are counted
locally with the project's single tokenizer (`memfaith.tokens`) and completion
tokens from the streamed text. That is the same accounting the rest of the audit
uses, which keeps the cost-per-correct-answer metric internally consistent.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from memfaith.tokens import count_tokens

# ---------------------------------------------------------------------------
# Credentials: the machine's own Trae login.
# ---------------------------------------------------------------------------
TRAE_HOME = os.path.expanduser("~")
# Written by the (unpublished) credential helper: a short-lived, machine-local token
# file. Kept project-local and gitignored; supply your own if you want to use this
# backend (see README "Model configuration").
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAE_STATE_PATH = os.path.join(_PROJECT_ROOT, ".cache", "trae_credential.json")
TRAE_STORAGE = os.path.join(os.environ.get("APPDATA", ""), "Trae CN", "User",
                            "globalStorage", "storage.json")

CN_CHAT_BASE = "https://trae-api-cn.mchost.guru"
AI_CHAT_BASE = "https://coresg-normal.trae.ai"
APP_ID = "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8"

# Wire ids that are callable through `solo_work_lite` / `solo_work_remote`.
# Chosen to give the audit three genuinely different model identities.
ROLE_MODELS = {
    "summarizer": "DeepSeek-V4-Flash-Official",
    "answerer": "glm-5.3-flash",
    "judge": "DeepSeek-V4-Pro-Official",
    "extractive": "DeepSeek-V4-Flash-Official",
}

FALLBACK_MODELS = [
    "DeepSeek-V4-Flash-Official",
    "DeepSeek-V4-Pro-Official",
    "glm-5.3-flash",
    "glm-5.3",
    "kimi-k3",
    "kimi-k2.6",
]

# Functions to try, in order. A model is only callable via one that lists it.
FUNCTIONS = ["solo_work_lite", "solo_work_remote", "chat_v3", "solo_coder", "solo_agent_remote"]


class TraeAuthError(RuntimeError):
    pass


def load_credential() -> dict:
    """Load the gateway credential from the local state file.

    The file is produced out-of-band (it carries a short-lived access token and is
    gitignored). Expected keys: ``token``, ``tokenExpireAt`` (epoch ms), ``region``,
    and optionally ``deviceId`` / ``machineId`` for the identity headers.
    """
    if not os.path.exists(TRAE_STATE_PATH):
        raise TraeAuthError(
            f"no credential at {TRAE_STATE_PATH}. This backend needs a token file; "
            f"supply your own (see README 'Model configuration'), or use an "
            f"OpenAI-compatible endpoint via MEMFAITH_BASE_URL + MEMFAITH_API_KEY."
        )
    with open(TRAE_STATE_PATH, "r", encoding="utf-8") as f:
        cred = json.load(f)
    token = cred.get("token")
    if not token:
        raise TraeAuthError(f"{TRAE_STATE_PATH} has no token")
    expires = cred.get("tokenExpireAt")
    if isinstance(expires, (int, float)) and expires / 1000 < time.time():
        raise TraeAuthError(
            f"credential at {TRAE_STATE_PATH} has expired; supply a fresh token file"
        )
    return cred


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
_EVENT_RE = re.compile(r"^event:(\S+)\s*$")


def parse_sse(text: str) -> dict[str, Any]:
    """Collapse an `llm_utils_chat` SSE stream into text + reasoning + errors."""
    visible: list[str] = []
    reasoning: list[str] = []
    errors: list[str] = []
    meta: dict[str, Any] = {}
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event == "output":
            if payload.get("response"):
                visible.append(payload["response"])
            if payload.get("reasoning_content"):
                reasoning.append(payload["reasoning_content"])
        elif event == "error":
            errors.append(json.dumps(payload, ensure_ascii=False)[:300])
        elif event == "metadata":
            meta.update(payload)
    return {
        "text": "".join(visible).strip(),
        "reasoning": "".join(reasoning),
        "errors": errors,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class TraeClient:
    """Minimal `LLMClient`-compatible client for the Trae SOLO gateway.

    Exposes ``chat(role, messages, **kw)`` and a ``ledger`` with ``records`` /
    ``totals()`` so it is a drop-in for `memfaith.llm.LLMClient` everywhere the
    audit uses one (compressors, evaluator, report).
    """

    def __init__(self, models: dict[str, str] | None = None, ledger: Any = None,
                 max_retries: int = 4, timeout: int = 180, verbose: bool = True,
                 use_cache: bool = False, cache_path: str | None = None):
        self.credential = load_credential()
        self.token = self.credential["token"]
        self.base = CN_CHAT_BASE if str(self.credential.get("region", "cn")).lower().startswith("cn") else AI_CHAT_BASE
        self.models = dict(ROLE_MODELS)
        if models:
            self.models.update(models)
        from memfaith.llm import DiskCache, Ledger

        self.ledger = ledger if ledger is not None else Ledger()
        self.cache = DiskCache(cache_path, enabled=bool(use_cache and cache_path))
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose
        self._working: dict[str, tuple[str, str]] = {}  # model -> (function, config_name)

    # -- headers -----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        trace = uuid.uuid4().hex + uuid.uuid4().hex
        trace = trace[:32]
        machine = self.credential.get("machineId", "unknown")
        device = self.credential.get("deviceId", "unknown")
        version = str(self.credential.get("appVersion", "2.3.93438"))
        return {
            "Authorization": f"Cloud-IDE-JWT {self.token}",
            "X-Ide-Token": self.token,
            "x-plugin-channel": "icube-ai",
            "User-Agent": f"Trae/{version}",
            "x-app-id": APP_ID,
            "x-machine-id": machine,
            "x-device-id": device,
            "x-device-type": "windows",
            "x-app-version": version,
            "x-ide-version": version,
            "x-app-version-code": "20260716",
            "x-ide-version-code": "20260716",
            "x-ide-version-type": "stable",
            "x-custom-trace-id": trace,
            "x-flow-traceparent": f"04-{trace}-{trace[:16]}-01",
            "request-traffic-type": "prod",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _body(messages: list[dict], model: str, function: str, max_tokens: int,
              temperature: float) -> dict:
        return {
            "messages": [
                {"role": m.get("role", "user"),
                 "content": [{"type": "text", "text": str(m.get("content", ""))}]}
                for m in messages
            ],
            "model": model,
            "config_name": model,
            "function": function,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    # -- transport ---------------------------------------------------------
    def _post(self, model: str, messages: list[dict], function: str,
              max_tokens: int, temperature: float) -> tuple[int, str]:
        url = f"{self.base}/api/agent/v3/llm_utils_chat"
        body = json.dumps(self._body(messages, model, function, max_tokens, temperature)).encode()
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "ignore")[:500]

    def _order(self, model: str) -> list[str]:
        """Functions to try for a model, remembering the one that worked."""
        if model in self._working:
            fn = self._working[model][0]
            return [fn] + [f for f in FUNCTIONS if f != fn]
        return list(FUNCTIONS)

    def chat(self, role: str, messages: list[dict], *, model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 900, tag: str = "",
             retries: int | None = None) -> str:
        model = model or self.models.get(role)
        if not model:
            raise RuntimeError(f"no model configured for role {role!r}")

        ck = None
        if self.cache.enabled:
            from memfaith.llm import DiskCache

            ck = DiskCache.key(model, messages, temperature, max_tokens)
            hit = self.cache.get(ck)
            if hit is not None:
                self.ledger.add(role=role, tag=tag, model=model, ok=True, cached=True,
                                prompt_tokens=0, completion_tokens=0, total_tokens=0, latency=0.0)
                return hit

        prompt_tokens = sum(count_tokens(str(m.get("content", ""))) for m in messages)
        attempts = retries if retries is not None else self.max_retries
        last = ""
        for attempt in range(attempts):
            for function in self._order(model):
                t0 = time.time()
                try:
                    status, raw = self._post(model, messages, function, max_tokens, temperature)
                    lat = round(time.time() - t0, 2)
                    if status != 200:
                        last = f"HTTP {status}: {raw[:180]}"
                        continue
                    parsed = parse_sse(raw)
                    if parsed["errors"]:
                        last = f"stream error: {parsed['errors'][0][:180]}"
                        continue
                    text = parsed["text"]
                    if not text and parsed["reasoning"]:
                        # Some models emit reasoning only when the budget is tight;
                        # a reasoning-only answer is not an answer.
                        last = "reasoning-only response (raise max_tokens)"
                        continue
                    if not text:
                        last = "empty response"
                        continue
                    self._working[model] = (function, model)
                    completion = count_tokens(text) + count_tokens(parsed["reasoning"])
                    self.ledger.add(role=role, tag=tag, model=model, ok=True, cached=False,
                                    prompt_tokens=prompt_tokens, completion_tokens=completion,
                                    total_tokens=prompt_tokens + completion, latency=lat,
                                    reasoning_tokens=count_tokens(parsed["reasoning"]),
                                    function=function)
                    if ck is not None:
                        self.cache.put(ck, text)
                    return text
                except Exception as e:  # noqa: BLE001
                    last = f"{type(e).__name__}: {str(e)[:150]}"
                    continue
            self.ledger.add(role=role, tag=tag, model=model, ok=False, cached=False,
                            prompt_tokens=0, completion_tokens=0, total_tokens=0, latency=0.0,
                            error=last)
            if attempt < attempts - 1:
                backoff = min(2 ** attempt * 2.0, 20)
                if self.verbose:
                    print(f"    [retry {attempt + 1}/{attempts - 1}] {role}: {last[:110]} "
                          f"-> sleep {backoff:.0f}s", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"role={role} model={model} failed: {last}")

    # -- misc --------------------------------------------------------------
    def flush(self) -> None:
        self.cache.flush()

    def usage(self) -> dict:
        return self.ledger.totals()


def make_client(verbose: bool = True, use_cache: bool = True) -> TraeClient:
    """Build a client using the project cache location."""
    from memfaith import config

    config.ensure_dirs()
    return TraeClient(
        verbose=verbose,
        use_cache=use_cache,
        cache_path=os.path.join(config.CACHE_DIR, "trae_llm.json") if use_cache else None,
    )
