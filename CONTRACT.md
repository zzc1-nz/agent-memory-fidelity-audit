# Interface Contract — Agent Memory-Compression Fidelity Audit

This file is the single source of truth for module boundaries. **Do not change
`memfaith/schema.py`, `memfaith/llm.py`, `memfaith/tokens.py`, or `memfaith/config.py`
without telling the Lead** — other modules are being written against them in parallel.

## Non-negotiable rules

1. **Token counting**: always use `memfaith.tokens.count_tokens` (cl100k_base). Never
   `len(text.split())`, never a model-reported count for the *context* size. Model-reported
   usage is only for *cost accounting* of LLM calls.
2. **All LLM calls** go through `memfaith.llm.LLMClient.chat(role, messages, ...)`. Roles are
   `summarizer`, `answerer`, `judge`, `extractive`. Never call `urllib` directly.
   This guarantees the token ledger is complete.
3. **Determinism**: dataset and probes are pure functions of `(seed, index)` — no LLM, no
   wall clock, no randomness outside `random.Random(seed)`. Re-running must reproduce
   byte-identical JSON.
4. **No network in unit tests.** Tests must run offline against fixtures.
5. Write only inside your assigned files (see ownership table). Use `write`/`edit`, not
   shell redirection.

## Directory ownership

| Path | Owner | Contents |
|---|---|---|
| `memfaith/dataset/*.py` | **dataset-eng** | long-history generator |
| `memfaith/probes/*.py` | **dataset-eng** | probe generator |
| `memfaith/compression/*.py` | **compress-eng** | the two compression strategies |
| `memfaith/metrics/*.py` | **eval-eng** | eval harness, judge, cost metrics, failure analysis |
| `memfaith/run.py`, `memfaith/report/*`, `README.md` | **Lead** | CLI, charts, docs |

Everyone may *read* anything. Nobody edits another's files; report conflicts to the Lead.

## 1. Dataset module — `memfaith/dataset/generator.py`

```python
def build_sessions(n: int, seed: int, turns_range: tuple[int, int] = (20, 30)) -> list[Session]
def save_sessions(sessions: list[Session], path: str) -> None      # JSONL of Session.to_dict()
def load_sessions(path: str) -> list[Session]
```

Requirements:
- Exactly `n` sessions, `session_id = "s{index:03d}"`, index starting at 1.
- Each session is 20–30 turns (`Turn.idx` 0-based, `role` in user/assistant/tool).
- Domain mix: roughly 40% `coding`, 30% `research`, 30% `daily`, deterministic.
- **High-frequency redundancy**: repeated status chatter, re-stated build steps, small talk
  that mentions nothing probe-worthy. This is what compression *should* remove.
  Aim for redundancy to be ≥45% of raw tokens.
- **Low-frequency critical facts**: each session carries 6–9 `Fact`s that probes depend on,
  each mentioned 1–3 times, deliberately varied along:
  - `salience`: `foreground` (user says "important/remember this") vs `background` (buried in
    an assistant paragraph)
  - `embedding`: `standalone` (own bullet/line) / `embedded` (mid-paragraph) /
    `deep_buried` (inside a long assistant answer, far from any cue)
  - `stability`: `stable` vs `superseded` (a later turn changes the value; the **latest**
    value is what probes expect — this is the "update/forgetting" axis borrowed from OpenMemX)
  - `kind`: `path`, `package`, `api_key_location`, `citation`, `dataset_source`, `date`,
    `preference`, `host`, `version`, `constraint`
- `Fact.value` must be a short canonical string (a path, a package name, an arXiv id, a date,
  a preference phrase) so the judge can match it literally.
- `Fact.mentions` must list the turn indices where the value appears; `first_turn` = min.
- `Session.full_tokens` = `count_tokens` of the fully rendered transcript
  (`"USER: ...\nASSISTANT: ...\n"` joined), which is what the "full context" condition sends.
- Add a module-level `render_transcript(session) -> str` used by both compressors and eval so
  everyone renders identically.

## 2. Probe module — `memfaith/probes/generator.py`

```python
def build_probes(sessions: list[Session], per_session: int = 10,
                 mix: tuple[int, int, int] = (3, 5, 2)) -> list[Probe]
def save_probes(probes: list[Probe], path: str) -> None
def load_probes(path: str) -> list[Probe]
```

- `mix` = (high_level, specific_fact, multi_hop) counts; sum must equal `per_session`.
- `probe_id = "{session_id}-p{index:02d}"`.
- **high_level**: answerable from a good summary (e.g. "What is the overall goal of this
  session and which stack was chosen?"). `fact_ids` lists ≥2 facts it summarizes.
- **specific_fact**: needs one exact value ("Which path holds the production API key?").
  `fact_ids` = exactly 1.
- **multi_hop**: needs ≥2 facts combined ("Which package did we settle on, and which host
  serves its registry?"). `hops` ≥ 2, `fact_ids` ≥ 2.
- **Crucially**: for `superseded` facts, probes must ask for the *current* value, and the
  reference answer must be the latest one. Include a `notes` field recording that the old
  value is a trap.
- `reference_answer` is a short literal string (the canonical value(s)), optionally with a
  brief clause. Keep it short: the judge compares against it.

## 3. Compression module — `memfaith/compression/`

```python
# memfaith/compression/base.py
class Compressor(Protocol):
    name: str
    def compress(self, session: Session, client: LLMClient) -> CompressionResult: ...

# memfaith/compression/summary.py
class HierarchicalSummaryCompressor:   name = "summary_hierarchical"
# memfaith/compression/extractive.py
class ExtractiveCompressor:            name = "extractive"
# memfaith/compression/__init__.py
def get_compressor(name: str) -> Compressor
```

- `CompressionResult.compressed_tokens = count_tokens(result.text)`.
- `original_tokens = session.full_tokens`.
- `ratio = compressed_tokens / original_tokens`.
- `cost_tokens` = **sum of prompt+completion tokens spent producing the compression**
  (read it from the ledger delta around your calls — the Lead will also cross-check).
- **Summary strategy**: hierarchical — chunk the transcript into windows, summarize each
  window (with a prompt that explicitly demands preserving paths/ids/dates/preferences),
  then summarize the summaries into a final memory. Record `meta["levels"]` and
  `meta["chunks"]`.
- **Extractive strategy**: self-contained, no model download. Score sentences
  (TF-IDF / TextRank-ish + numeric/entity/path boosters) and keep a budgeted subset in
  original order. Also implement `meta["method"]`. If `llmlingua` happens to be importable,
  you may add an adapter behind `MEMFAITH_USE_LLMLINGUA=1`, but it must degrade gracefully.
- Both must be **budget-aware**: hit the target ratio in `RunConfig` (≈0.25) rather than
  producing whatever length comes out. Truncating the final summary to the budget is allowed
  but must be recorded in `meta["truncated"]`.

## 4. Metrics / eval module — `memfaith/metrics/`

```python
# memfaith/metrics/judge.py
def judge_answer(client: LLMClient, probe: Probe, answer: str) -> tuple[bool | None, str, float | None]
# memfaith/metrics/evaluate.py
def run_matrix(sessions, probes, compressors, client, cfg, out_dir) -> dict
# memfaith/metrics/cost.py
def cost_per_correct_answer(...) -> dict
# memfaith/metrics/failures.py
def analyze_failures(answers, probes, sessions, compressions) -> dict
```

- The judge must be a **different model** than the answerer (enforced by `config.MODELS`).
  Prompt it to return strict JSON `{"verdict":"correct"|"incorrect"|"partial","confidence":0..1}`;
  treat `partial` as incorrect for the headline metric but report it separately.
- Conditions per probe: `full` (whole transcript) and `compressed` (strategy output).
- Headline metric, per strategy:
  - `total_compressed_tokens` = sum of `compressed_tokens` over sessions
    **plus** the tokens spent producing the compression (`cost_tokens`) — state clearly
    which variant you report; report both (`amortized` vs `one_shot`).
  - `correct_count` = number of probes answered correctly under the compressed condition.
  - `cost_per_correct_answer = total_compressed_tokens / correct_count`.
  - Baseline: same formula with `full_context` tokens and its correct count.
- `failures.py` must join lost facts (facts whose probes flipped correct→incorrect) against
  their attributes (`frequency`, `salience`, `embedding`, `stability`, `kind`) and emit
  aggregate stats + the individual cases with the compressed text snippet that should have
  contained them.

## 5. Reporting — Lead

`memfaith/run.py` orchestrates: build → compress → probe → evaluate → report.
Outputs into `results/`: `sessions.jsonl`, `probes.jsonl`, `compressions.jsonl`,
`answers.jsonl`, `metrics.json`, `failures.json`, `report.md`, charts under `results/figures/`.

## Environment facts (already verified)

- Python 3.14.5, no pip install possible (sandbox blocks wheel unpack in temp). Use vendored
  `vendor/tiktoken` via `memfaith.tokens`.
- LLM gateway: keyless `https://api.kilo.ai/api/openrouter/v1`, verified working.
- Role models: summarizer `kilo-auto/free`, answerer `nvidia/nemotron-3.5-super-vl:free`,
  judge `nvidia/nemotron-3-ultra-550b-a55b:free`.
- The gateway is flaky (~10–20% 400s) and slow (3–30s). `LLMClient` already retries with
  backoff and caches to disk. **Always pass a real `max_tokens`** (≥300 for judge/answer,
  ≥900 for summarizer): reasoning models burn budget on hidden tokens and return empty
  content if it is too small.
