# Agent Memory-Compression Fidelity Audit

> **Compression ratio is not the goal. Cost per correct answer is.**

An audit benchmark that asks the question the compression literature leaves open.
OpenMemX reports a 20-40% prompt reduction with LLMLingua-2 extractive compression
"without losing semantic coherence"; AgentMem reports 405x context compression at
~873 tokens/session. Neither answers the question that decides whether a compressed
memory is usable:

> **What was compressed away - redundancy, or the facts the task needs?**

This benchmark measures that directly. It builds synthetic long agent-user histories
that deliberately contain both classes of content, compresses them with two
paradigms, probes what survived, and reports the metric that matters:

```
cost per correct answer = total tokens spent / probes answered correctly
```

---

## Headline result (30 sessions, 300 probes, 1.66M tokens, real models)

| Arm | Compression ratio | Accuracy | Correct | Tokens / correct answer | vs baseline |
|---|---|---|---|---|---|
| full context (no compression) | 100.0% | 67.3% | 202/300 | **2,061** | - |
| hierarchical summary (LLM) | 22.7% | 63.0% | 189/300 | **1,877** | -8.9% |
| extractive (deterministic) | 24.7% | 51.0% | 153/300 | **672** | **-67.4%** |

**Three findings the ratio-only framing misses.**

1. **The two paradigms reach the same ratio and differ by 7x in cost.** Both land at
   ~23-25%, so "which compresses more?" has no discriminating power here. But the
   summary arm must *spend* 260,121 tokens to produce its summaries while the
   extractive arm spends 0, and that one difference decides the economics.

2. **Abstraction is a capital cost, selection is an operating cost.** The summary arm
   saves ~10,700 context tokens per session but spends 260,121 to build all 30
   summaries. It only amortizes if each compressed memory is reused **~24+ times** -
   a property of the *workload*, not of the compressor. For one-shot probing it is
   strictly worse than doing nothing.

3. **Some "losses" are not losses at all.** Of 84 probes that flipped correct ->
   incorrect, **17 depend entirely on facts that are still present verbatim** in the
   compressed memory. The memory held both the stale and the current value of a
   superseded fact; the literal string survived, the reader had no cue which was
   current, and it picked the wrong one. A verbatim-retention metric scores these as
   perfect; a cost-per-correct-answer metric catches them.

## Why the existing metrics are insufficient

| Claim | What it measures | What it hides |
|---|---|---|
| "20-40% fewer prompt tokens" | input length | whether the removed tokens were load-bearing |
| "405x context compression" | length ratio | whether any downstream task still succeeds |
| "semantic coherence preserved" | human/LLM plausibility | whether a *specific fact* survived verbatim |
| "873 tokens/session" | storage footprint | whether the session is still answerable |

A summary can be perfectly coherent and still drop the one path, port, or date the
next task needs. Coherence is a property of the *summary*; fidelity is a property of
the *summary relative to the task*.

## Design

### 1. Synthetic long histories with controlled redundancy and criticality

Each session is 20-30 turns of agent-user interaction across three task families
(coding / research / daily). Two content classes are planted deliberately:

- **Redundant filler** (measured mean 45.9% of raw tokens): restated build steps,
  status chatter, re-confirmations. *This is what compression should remove.*
- **Critical facts** (6-9 per session, 220 total, mentioned 1-3 times): the facts
  probes depend on, annotated along four axes that predict survivability:

  | Axis | Values | Hypothesis |
  |---|---|---|
  | `frequency` | 1 / 2 / 3+ mentions | low frequency -> dropped first |
  | `salience` | foreground / background | background -> dropped first |
  | `embedding` | standalone / embedded / deep_buried | deep_buried -> dropped first |
  | `stability` | stable / superseded | superseded -> the *old* value may win |

  The `superseded` axis is the "forgetting" mechanism borrowed from OpenMemX's layered
  abstraction: a later turn overwrites an earlier value. A compressor that keeps the
  earlier mention produces a confidently wrong answer - the worst failure mode, and
  the one this benchmark was built to catch.

### 2. Two compression arms

| Arm | Mechanism | Analogue |
|---|---|---|
| `summary_hierarchical` | map-reduce LLM summarization: chunk -> summarize -> reduce, with an explicit preserve-these-identifiers instruction | OpenMemX layered abstraction / AgentMem cognitive-stack compression |
| `extractive` | deterministic sentence scoring (TF-IDF + identifier boosters + redundancy penalty + recency), budgeted subset in original order | LLMLingua-2-style extractive compression, self-contained |

Both are **budget-aware** (target ~25% of original tokens) so the comparison is at
matched compression rather than at whatever length each happens to emit.

### 3. Three probe families

| Type | Count/session | Tests | Expected |
|---|---|---|---|
| `high_level` | 3 | gist ("what was the goal, which stack won") | summary should pass |
| `specific_fact` | 5 | one exact literal (path, port, date, citation) | compression may lose |
| `multi_hop` | 2 | combine >= 2 compressed facts | hardest |

Measured loss rate by type: high-level 22.8%, multi-hop 11.7%, specific-fact 9.7% -
note that the type a *summary* is supposed to handle best is where it loses most,
because a gist answer must reproduce several literals at once.

Each probe runs in two conditions (full context, compressed) and is graded by an LLM
judge that is a **different model** from the answerer (self-bias control). Judge
calibration on hand-labelled pairs: 14/14 agreement, 0/8 false accepts on the
dangerous direction (accepting a plausible-but-wrong answer).

### 4. The metric that matters

```
cost_per_correct_answer = (context_tokens + compression_cost_tokens) / n_correct
```

reported for the baseline and each arm, plus a **breakeven** derivation:

```
compression pays off  <=>  accuracy_retention > compression_ratio
```

If you cut tokens by 75% but retain only 60% of the answers, you *increased* the cost
of every correct answer by ~1.4x.

### 5. Failure analysis

Every probe that flips correct -> incorrect is joined back to its ground-truth facts
and aggregated by the four axes, with a per-arm breakdown (pooling the arms hides the
finding: a fact counts as survived if *any* arm kept it, so pooled rates sit near 1.0).
Individual cases carry a verbatim-survival check, which is what separates true
information loss from value confusion.

**Measured survival by arm** (verbatim, of probed facts):

| Attribute | Value | extractive | summary |
|---|---|---|---|
| embedding | deep_buried | 78.2% | 96.2% |
| embedding | embedded | 88.1% | 92.9% |
| salience | background | 78.5% | 92.6% |
| salience | foreground | 91.5% | 97.9% |
| stability | superseded | 83.3% | 90.0% |

The extraction arm drops buried/background facts at ~4x the rate of the summary arm,
yet still wins on cost per correct answer - because it spends nothing to decide.

---

## Quickstart

```bash
python scripts/bootstrap.py                      # vendor tiktoken (one-time)
python scripts/test.py                           # offline unit tests
python -m memfaith.run --n-sessions 3 --probes 6 --tag smoke    # ~5 min smoke run
python -m memfaith.run --n-sessions 30 --probes 10 --tag full   # full experiment
python -m memfaith.run --stage report --tag full                # rebuild report only
```

Outputs land in `results/`:

```
results/
  sessions.jsonl      probes.jsonl      compressions.jsonl
  answers.jsonl       metrics.json      failures.json
  usage.json          report.md         figures/*.png
```

## Model configuration

Three distinct model identities are required so the judge never grades its own
generation. Two backends are supported, selected automatically:

1. **Any OpenAI-compatible endpoint** - set `MEMFAITH_BASE_URL` + `MEMFAITH_API_KEY`:

   ```bash
   export MEMFAITH_BASE_URL=https://api.openai.com/v1
   export MEMFAITH_API_KEY=sk-...
   export MEMFAITH_MODEL_SUMMARIZER=gpt-4o-mini
   export MEMFAITH_MODEL_ANSWERER=gpt-4o-mini
   export MEMFAITH_MODEL_JUDGE=gpt-4o            # must differ from answerer
   ```

2. **A private gateway adapter** - `memfaith/trae_client.py` targets an
   OpenAI-incompatible gateway that speaks an SSE variant (`Cloud-IDE-JWT` auth,
   `config_name` wire ids, per-install identity headers). It is included because the
   experiments reported above ran through it, and it demonstrates adapting an arbitrary
   provider to the same `chat()` surface. The credential-acquisition helper for that
   specific local login is **deliberately not published** (see `local/README.md`);
   supply your own credential if you want to exercise this path.

   Role models used for the reported run:

   | Role | Model |
   |---|---|
   | summarizer | `DeepSeek-V4-Flash-Official` |
   | answerer | `glm-5.3-flash` |
   | judge | `DeepSeek-V4-Pro-Official` |

Token counting is always `cl100k_base` via the vendored `tiktoken`, independent of
whichever model serves the call, so the accounting is comparable across backends.

## Repo layout

```
memfaith/
  schema.py            shared data contracts (Session / Fact / Probe / Answer / ...)
  tokens.py            single source of truth for token counting
  llm.py               OpenAI-compatible client: retrying, disk-cached, token-accounted
  trae_client.py       Trae SOLO gateway adapter (same chat() surface)
  config.py            model roles + run size
  dataset/generator.py synthetic long histories with annotated ground truth
  probes/generator.py  three probe families
  compression/         hierarchical summary + extractive arms
  metrics/             judge, eval matrix, cost metric, failure analysis
  report/build.py      Markdown report + charts
  run.py               CLI orchestration (resumable)
scripts/               bootstrap, credential refresh, judge calibration, test runner
tests/                 132 offline unit tests
```

## Limitations

- Synthetic histories are controlled but not real; they isolate the variables the
  audit needs to measure at the cost of some messiness.
- LLM-judged correctness is noisy at the margin; raw judge output is retained so
  verdicts can be re-audited, and the judge is calibrated against hand labels.
- Two arms, not the full design space. LLMLingua-2 is wired behind a flag but not the
  default (no model download in this environment).
- `cl100k_base` is a neutral proxy tokenizer, not each model's native one.
- The Trae backend is rate-limited by the account; the run takes ~80 minutes at
  concurrency 3, and the client retries with backoff.
