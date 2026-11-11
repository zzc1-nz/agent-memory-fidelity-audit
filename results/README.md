# results/ - the experiment this repo is about

These files are the output of one full run (30 sessions, 300 probes, ~1.66M tokens)
and are committed on purpose: the numbers in the top-level README should be checkable
against the data that produced them, not taken on faith.

## What is committed

| File | What it is |
|---|---|
| `metrics.json` | The three-dimensional comparison (ratio / accuracy / cost per correct answer), the breakeven derivation, and per-probe-type breakdowns. |
| `failures.json` | Every probe that flipped correct -> incorrect, joined to its ground-truth facts, with per-arm survival aggregates. This is the backing data for the qualitative failure analysis. |
| `report.md` | The generated report (Markdown). |
| `figures/*.png` | The four charts referenced by the report. |
| `usage.json` | Actual token spend of the run, by model role. |

## What is not committed, and why

| File | Reason |
|---|---|
| `sessions.jsonl`, `probes.jsonl` (in `data/`) | Fully deterministic from `--seed`; regenerated in seconds by `python -m memfaith.run --stage build`. No information is lost by omitting them. |
| `answers.jsonl` (~420 KB) | 900 raw model answers. Regenerable, but **costs real tokens** to rebuild — so it is omitted to keep the repo light rather than because it is sensitive. |
| `compressions.jsonl`, `run_full.log`, `eval_summary.json` | Intermediate logs with no analytical value beyond what `metrics.json` / `failures.json` already contain. |

## Reproducing exactly

```bash
python scripts/bootstrap.py
python -m memfaith.run --n-sessions 30 --probes 10 --tag full
```

The dataset and probe generators are pure functions of `(n, seed)`, so the same
`--seed` reproduces byte-identical sessions and probes. Model answers will differ
slightly on a re-run (sampling, and a different serving model if you point the client
at a different endpoint); the committed `metrics.json` is the record of the run
reported in the README.
