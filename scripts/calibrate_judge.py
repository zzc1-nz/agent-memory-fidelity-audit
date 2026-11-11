"""Judge calibration probe: is the LLM judge trustworthy enough to build a metric on?

The audit's headline number is only as good as the judge. Before trusting it we
measure three things on hand-labelled pairs:

  1. agreement with human labels (accuracy)
  2. false-positive rate: does it accept a plausible-but-wrong answer?
     (the dangerous direction for a compression audit — it would hide lost facts)
  3. false-negative rate: does it reject a correct answer phrased differently?

Usage: python scripts/calibrate_judge.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memfaith import config  # noqa: E402
from memfaith.metrics.judge import judge_answer  # noqa: E402
from memfaith.schema import Probe  # noqa: E402


def build_client():
    """Same backend selection as `memfaith.run.make_client`."""
    config.ensure_dirs()
    cache = os.path.join(config.CACHE_DIR, "judge_calibration.json")
    if config.API_KEY or os.environ.get("MEMFAITH_FORCE_OPENAI"):
        from memfaith.llm import LLMClient

        return LLMClient(base_url=config.BASE_URL, api_key=config.API_KEY,
                         models=config.MODELS, cache_path=cache)
    from memfaith.trae_client import TraeClient

    return TraeClient(use_cache=True, cache_path=cache)

# (reference, candidate, expected_correct, rationale)
CASES = [
    # --- should be judged CORRECT ---
    ("/srv/app/.env.prod", "The production key is in /srv/app/.env.prod.", True,
     "verbatim containment"),
    ("6432", "Port 6432.", True, "verbatim numeric"),
    ("arXiv:2405.12904v3", "We cite arXiv:2405.12904v3 by Okafor et al.", True,
     "id inside a sentence"),
    ("uv", "The team settled on uv as the package manager.", True, "short literal in prose"),
    ("2026-12-04", "The deadline is 2026-12-04.", True, "date"),
    # --- should be judged INCORRECT (plausible but wrong) ---
    ("/srv/app/.env.prod", "The production key is in /srv/app/.env.staging.", False,
     "near-miss path — the classic compression failure"),
    ("6432", "Port 5432.", False, "transposed digits"),
    ("arXiv:2405.12904v3", "We cite arXiv:2405.12904v2.", False, "version suffix differs"),
    ("uv", "The team settled on pip.", False, "wrong tool, right category"),
    ("2026-12-04", "The deadline is 2026-12-14.", False, "adjacent date"),
    # --- superseded trap: the OLD value must not count ---
    ("/etc/svc/credentials", "The key moved to /etc/svc/credentials.", True,
     "current value"),
    ("/etc/svc/credentials", "The key is at ~/.config/svc/token.env.", False,
     "STALE value — must be rejected"),
    # --- evasive / unhelpful answers must not count ---
    ("6432", "I don't have enough information to answer that.", False, "refusal"),
    ("6432", "It was probably some port in the 6000s.", False, "hedged guess"),
]


def main() -> int:
    client = build_client()
    probe = Probe(
        probe_id="calib", session_id="calib", kind="specific_fact", question="q",
        reference_answer="", fact_ids=[], domain="coding",
    )
    tp = tn = fp = fn = 0
    rows = []
    for ref, cand, expected, why in CASES:
        probe.reference_answer = ref
        verdict, raw, conf = judge_answer(client, probe, cand)
        got = bool(verdict)
        ok = got == expected
        if expected and got:
            tp += 1
        elif not expected and not got:
            tn += 1
        elif not expected and got:
            fp += 1
        else:
            fn += 1
        rows.append((ok, ref, cand, expected, got, conf, why))
        print(f"{'OK ' if ok else 'MISS'} exp={expected!s:5} got={str(got):5} "
              f"conf={conf} ref={ref!r} cand={cand[:52]!r}")

    n = len(CASES)
    acc = (tp + tn) / n
    print(f"\nagreement: {tp + tn}/{n} = {acc:.1%}")
    print(f"false-positive (accepted a wrong answer): {fp}/{fp + tn or 1} "
          f"of negatives  <- dangerous direction")
    print(f"false-negative (rejected a right answer): {fn}/{tp + fn or 1} of positives")
    print(f"cached calls: {sum(1 for r in client.ledger.records if r.get('cached'))}/{n}")
    client.flush()
    return 0 if acc >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
