"""End-to-end orchestration for the memory-compression fidelity audit.

Pipeline
--------
  build dataset  ->  generate probes  ->  compress (2 strategies)
                 ->  answer + judge (full vs compressed)
                 ->  metrics (cost per correct answer)  ->  failure analysis
                 ->  report

Everything is resumable: `answers.jsonl` is appended per completed cell, and the
LLM layer caches identical prompts to disk, so a crash or a flaky gateway costs
time but never correctness.

Usage
-----
  python -m memfaith.run --n-sessions 30 --probes 10 --tag full
  python -m memfaith.run --n-sessions 3 --probes 6 --tag smoke --no-cache
  python -m memfaith.run --stage report        # rebuild report from existing results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from memfaith import config
from memfaith.compression import get_compressor
from memfaith.dataset.generator import build_sessions, load_sessions, render_transcript, save_sessions
from memfaith.llm import LLMClient
from memfaith.metrics.cost import compute_metrics
from memfaith.metrics.evaluate import run_matrix
from memfaith.metrics.failures import analyze_failures
from memfaith.probes.generator import build_probes, load_probes, save_probes
from memfaith.schema import CompressionResult, read_jsonl, write_jsonl
from memfaith.tokens import count_tokens


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scale_mix(base: tuple[int, int, int], total: int) -> tuple[int, int, int]:
    """Rescale a probe mix to sum exactly to ``total``, keeping the proportions.

    Largest-remainder rounding, with the high-level share floored at 1 when the
    total allows it: a run with no gist probes could not tell "compression lost
    the specifics" apart from "compression lost everything".
    """
    if total <= 0:
        return (0, 0, 0)
    base_sum = sum(base)
    raw = [b * total / base_sum for b in base]
    out = [int(x) for x in raw]
    remainder = total - sum(out)
    # Hand out the leftover slots to the largest fractional parts.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:remainder]:
        out[i] += 1
    if total >= 3 and out[0] == 0:
        out[0], out[1] = 1, out[1] - 1 if out[1] > 1 else out[1]
        while sum(out) != total:  # keep the invariant even in degenerate cases
            out[1] += 1
    return (out[0], out[1], out[2])


def make_client(cfg: config.RunConfig, cache_name: str = "llm"):
    """Build the LLM client for this run.

    Backend selection, in priority order:
      1. ``MEMFAITH_BASE_URL`` + ``MEMFAITH_API_KEY`` -> any OpenAI-compatible endpoint
      2. the machine's own Trae login -> ``memfaith.trae_client`` (the default here)
    Both expose the same ``chat(role, messages, ...)`` surface and the same ledger,
    so every downstream metric is backend-independent.
    """
    config.ensure_dirs()
    cache = config.cache_path(cache_name) if cfg.use_cache else None
    if config.API_KEY or os.environ.get("MEMFAITH_FORCE_OPENAI"):
        return LLMClient(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            models=config.MODELS,
            cache_path=cache,
            use_cache=cfg.use_cache,
        )
    from memfaith.trae_client import TraeClient

    return TraeClient(
        use_cache=cfg.use_cache,
        cache_path=cache,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Stage 1+2: dataset and probes (no LLM, deterministic)
# ---------------------------------------------------------------------------
def stage_build(cfg: config.RunConfig, data_dir: str) -> tuple[list, list]:
    sess_path = os.path.join(data_dir, "sessions.jsonl")
    probe_path = os.path.join(data_dir, "probes.jsonl")

    sessions = build_sessions(cfg.n_sessions, cfg.seed, (cfg.turns_min, cfg.turns_max))
    save_sessions(sessions, sess_path)
    probes = build_probes(sessions, cfg.probes_per_session, cfg.probe_mix)
    save_probes(probes, probe_path)

    total_tokens = sum(s.full_tokens for s in sessions)
    _log(f"dataset: {len(sessions)} sessions, {sum(len(s.turns) for s in sessions)} turns, "
         f"{total_tokens:,} tokens total (avg {total_tokens // max(1, len(sessions)):,}/session)")
    kinds: dict[str, int] = {}
    for p in probes:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    _log(f"probes: {len(probes)} total {kinds}")
    return sessions, probes


# ---------------------------------------------------------------------------
# Stage 3: compression
# ---------------------------------------------------------------------------
def stage_compress(sessions, cfg: config.RunConfig, client: LLMClient,
                   results_dir: str) -> dict[str, list[CompressionResult]]:
    out: dict[str, list[CompressionResult]] = {}
    for name in cfg.strategies:
        comp = get_compressor(name)
        rows: list[CompressionResult] = []
        t0 = time.time()
        for i, s in enumerate(sessions, 1):
            try:
                r = comp.compress(s, client)
            except Exception as e:  # noqa: BLE001
                _log(f"  !! {name} failed on {s.session_id}: {type(e).__name__}: {e}")
                r = CompressionResult(
                    session_id=s.session_id, strategy=name, text="",
                    compressed_tokens=0, original_tokens=s.full_tokens, ratio=0.0,
                    cost_tokens=0, meta={"error": f"{type(e).__name__}: {e}"},
                )
            rows.append(r)
            if i % 5 == 0 or i == len(sessions):
                _log(f"  {name}: {i}/{len(sessions)} sessions "
                     f"(mean ratio {sum(x.ratio for x in rows) / len(rows):.3f})")
        client.flush()
        out[name] = rows
        _log(f"{name}: done in {time.time() - t0:.0f}s, "
             f"mean ratio {sum(r.ratio for r in rows) / max(1, len(rows)):.3f}, "
             f"compression cost {sum(r.cost_tokens for r in rows):,} tokens")

    write_jsonl(os.path.join(results_dir, "compressions.jsonl"),
                [r.to_dict() for rows in out.values() for r in rows])
    return out


# ---------------------------------------------------------------------------
# Stage 4: answer + judge
# ---------------------------------------------------------------------------
def stage_evaluate(sessions, probes, compressions, cfg: config.RunConfig,
                   client: LLMClient, results_dir: str) -> list[dict]:
    answers = run_matrix(sessions, probes, compressions, client, cfg, results_dir)
    client.flush()
    _log(f"evaluation: {len(answers)} answers recorded")
    return answers


# ---------------------------------------------------------------------------
# Stage 5+6: metrics, failures, report
# ---------------------------------------------------------------------------
def stage_report(sessions, probes, compressions, answers, cfg: config.RunConfig,
                 results_dir: str) -> dict:
    metrics = compute_metrics(answers, probes, sessions, compressions, cfg)
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    failures = analyze_failures(answers, probes, sessions, compressions)
    with open(os.path.join(results_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    from memfaith.report.build import build_report

    report_path = build_report(sessions, probes, compressions, answers, metrics, failures,
                               cfg, results_dir)
    _log(f"report -> {report_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent memory-compression fidelity audit")
    ap.add_argument("--n-sessions", type=int, default=30)
    ap.add_argument("--probes", type=int, default=10, help="probes per session")
    ap.add_argument("--turns-min", type=int, default=20)
    ap.add_argument("--turns-max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--stage", choices=["all", "build", "compress", "eval", "report"],
                    default="all")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    cfg = config.RunConfig(
        n_sessions=args.n_sessions, probes_per_session=args.probes,
        turns_min=args.turns_min, turns_max=args.turns_max, seed=args.seed,
        concurrency=args.concurrency, tag=args.tag, use_cache=not args.no_cache,
    )
    # The probe mix is (high_level, specific_fact, multi_hop) and must sum to the
    # requested per-session count. Rescale the default 3:5:2 split so `--probes N`
    # always works instead of failing the mix validation downstream.
    if sum(cfg.probe_mix) != cfg.probes_per_session:
        cfg.probe_mix = scale_mix(config.RunConfig().probe_mix, cfg.probes_per_session)
    _log(f"probe mix: {cfg.probe_mix} (sum={sum(cfg.probe_mix)})")
    config.ensure_dirs()
    results_dir = os.path.join(config.RESULTS_DIR, cfg.tag) if cfg.tag != "full" else config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    _log(f"run tag={cfg.tag} sessions={cfg.n_sessions} probes/session={cfg.probes_per_session} "
         f"-> {results_dir}")

    sess_path = os.path.join(config.DATA_DIR, "sessions.jsonl")
    probe_path = os.path.join(config.DATA_DIR, "probes.jsonl")
    comp_path = os.path.join(results_dir, "compressions.jsonl")
    ans_path = os.path.join(results_dir, "answers.jsonl")

    if args.stage in ("all", "build"):
        sessions, probes = stage_build(cfg, config.DATA_DIR)
    elif os.path.exists(sess_path):
        sessions, probes = load_sessions(sess_path), load_probes(probe_path)
        _log(f"loaded {len(sessions)} sessions, {len(probes)} probes from disk")
    else:
        sessions, probes = stage_build(cfg, config.DATA_DIR)

    client = make_client(cfg)

    compressions: dict[str, list[CompressionResult]] = {}
    if args.stage in ("all", "compress"):
        compressions = stage_compress(sessions, cfg, client, results_dir)
    elif os.path.exists(comp_path):
        for row in read_jsonl(comp_path):
            compressions.setdefault(row["strategy"], []).append(CompressionResult(**row))
        _log(f"loaded {sum(len(v) for v in compressions.values())} compression results")

    answers: list[dict] = []
    if args.stage in ("all", "eval"):
        answers = stage_evaluate(sessions, probes, compressions, cfg, client, results_dir)
    elif os.path.exists(ans_path):
        answers = read_jsonl(ans_path)
        _log(f"loaded {len(answers)} answers")

    if args.stage in ("all", "eval", "report"):
        if answers and compressions:
            stage_report(sessions, probes, compressions, answers, cfg, results_dir)
        else:
            _log("nothing to report (missing answers or compressions)")

    client.flush()
    usage = client.usage()

    # A `--stage report` rebuild does no LLM work, so its fresh ledger is empty.
    # Preserve the spend of the run that actually produced the data rather than
    # overwriting it with zeros.
    usage_path = os.path.join(results_dir, "usage.json")
    if usage["grand_total"]["calls"] == 0 and os.path.exists(usage_path):
        try:
            with open(usage_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            if previous.get("grand_total", {}).get("calls"):
                _log(f"no new LLM calls; keeping recorded usage from the original run "
                     f"({previous['grand_total']['total_tokens']:,} tokens)")
                usage = previous
        except (OSError, json.JSONDecodeError):
            pass

    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)
    g = usage["grand_total"]
    _log(f"token usage: {g['total_tokens']:,} total "
         f"({g['prompt_tokens']:,} prompt + {g['completion_tokens']:,} completion), "
         f"{g['calls']} calls, {g['failures']} failed attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
