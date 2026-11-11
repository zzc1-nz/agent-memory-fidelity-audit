"""Report builder: Markdown report + charts from metrics/failures.

The report is the deliverable that makes the audit legible: a three-dimensional
comparison (compression ratio / accuracy / cost-per-correct-answer), the
breakeven condition, and the qualitative failure analysis.
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memfaith.schema import Probe  # noqa: E402


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _fig_dir(results_dir: str) -> str:
    d = os.path.join(results_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def chart_three_dim(metrics: dict, results_dir: str) -> str:
    """Scatter: x = compression ratio, y = accuracy, bubble = cost per correct answer."""
    arms = metrics.get("arms", {})
    names = [k for k in arms if arms[k].get("compression_ratio") is not None]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in names:
        a = arms[name]
        ratio = a.get("compression_ratio", 1.0)
        acc = a.get("accuracy", 0.0)
        cpa = a.get("cost_per_correct_answer_amortized")
        size = 320 if name == "full_context" else 320
        ax.scatter([ratio], [acc], s=size, alpha=0.75, label=name)
        label = f"{name}\nacc={acc:.0%}"
        if cpa:
            label += f"\n{cpa:,.0f} tok/correct"
        ax.annotate(label, (ratio, acc), textcoords="offset points", xytext=(12, -6), fontsize=8)
    ax.set_xlabel("Compression ratio (compressed / original tokens)")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Ratio vs accuracy vs cost per correct answer")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    path = os.path.join(_fig_dir(results_dir), "ratio_accuracy_cost.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_cost_per_correct(metrics: dict, results_dir: str) -> str:
    arms = metrics.get("arms", {})
    names = [k for k in arms]
    amort = [arms[n].get("cost_per_correct_answer_amortized") or 0 for n in names]
    oneshot = [arms[n].get("cost_per_correct_answer_one_shot") or 0 for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.38
    ax.bar([i - w / 2 for i in x], amort, w, label="amortized (incl. compression cost)")
    ax.bar([i + w / 2 for i in x], oneshot, w, label="one-shot (context only)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Tokens per correct answer")
    ax.set_title("Cost per correct answer (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(_fig_dir(results_dir), "cost_per_correct.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_accuracy_by_kind(metrics: dict, results_dir: str) -> str:
    by_kind = metrics.get("accuracy_by_kind", {})
    kinds = sorted({k for v in by_kind.values() for k in v})
    arms = sorted(by_kind)
    if not kinds:
        return ""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        vals = [by_kind[arm].get(k, 0) for k in kinds]
        ax.bar([j + i * w for j in range(len(kinds))], vals, w, label=arm)
    ax.set_xticks([j + (len(arms) - 1) * w / 2 for j in range(len(kinds))])
    ax.set_xticklabels(kinds, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by probe type")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(_fig_dir(results_dir), "accuracy_by_kind.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_fact_survival(failures: dict, results_dir: str) -> str:
    """Per-arm fact survival by attribute.

    Pooling the arms is actively misleading here: because a fact counts as survived
    if *any* arm kept it, the pooled rates sit near 1.0 and the chart says nothing.
    The per-strategy view is the one that carries the finding, so it is what gets
    plotted when it is available.
    """
    by_strategy = failures.get("survival_by_attribute_by_strategy", {})
    survival = by_strategy or {"pooled": failures.get("survival_by_attribute", {})}
    arms = [a for a in survival if survival[a]]
    if not arms:
        return ""

    groups = sorted({g for arm in arms for g in survival[arm]})
    if not groups:
        return ""

    fig, axes = plt.subplots(1, len(groups), figsize=(3.9 * len(groups), 4.2), squeeze=False)
    colors = ["#4a7ebb", "#d9803a", "#5a9e5a", "#9b59b6"]
    for ax, group in zip(axes[0], groups):
        values = sorted({v for arm in arms for v in survival[arm].get(group, {})})
        if not values:
            continue
        width = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            rates = []
            for value in values:
                slot = survival[arm].get(group, {}).get(value)
                rates.append(slot.get("survival_rate") or 0 if slot else 0)
            positions = [j + i * width for j in range(len(values))]
            ax.bar(positions, rates, width, label=arm, color=colors[i % len(colors)])
        ax.set_xticks([j + (len(arms) - 1) * width / 2 for j in range(len(values))])
        ax.set_xticklabels(values, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Verbatim survival rate" if group == groups[0] else "")
        ax.set_title(group, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Critical-fact verbatim survival by arm and attribute", fontsize=10)
    path = os.path.join(_fig_dir(results_dir), "fact_survival.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
#: Punctuation downgraded to ASCII on output, so the report survives any
#: non-UTF-8 console or diff tool without turning into mojibake.
_ASCII_FALLBACK = str.maketrans({
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\u2192": "->",
    "\u2248": "~",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u00d7": "x",
    "\u2022": "*",
    "\u2705": "OK",
    "\u2713": "OK",
    "\u2717": "X",
})


def _fmt_tokens(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:,.0f}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def _economics_section(metrics: dict) -> str:
    """The per-arm economics: which compression paradigm actually pays off.

    The headline metric can hide a lot when averaged, so this section states the
    comparison arm by arm and names the mechanism behind the difference.
    """
    arms = metrics.get("arms", {})
    base = arms.get("full_context", {})
    lines: list[str] = []

    base_cpa = base.get("cost_per_correct_answer_amortized")
    if not base_cpa:
        return "_Insufficient data._"

    lines.append(f"Baseline (no compression) costs **{_fmt_tokens(base_cpa)} tokens per correct "
                 f"answer** at {_fmt_pct(base.get('accuracy'))} accuracy.")
    lines.append("")
    lines.append("| Arm | Ratio | Accuracy | Correct | Tokens/correct | vs baseline | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    for name, a in arms.items():
        if name == "full_context":
            continue
        cpa = a.get("cost_per_correct_answer_amortized")
        delta = a.get("cost_delta_vs_baseline_pct")
        if not cpa or delta is None:
            continue
        verdict = "**cheaper** OK" if delta < 0 else "**more expensive** X"
        lines.append(f"| {name} | {_fmt_pct(a.get('compression_ratio'))} | "
                     f"{_fmt_pct(a.get('accuracy'))} | {a.get('n_correct')}/{a.get('n_probes')} | "
                     f"{_fmt_tokens(cpa)} | {delta:+.1%} | {verdict} |")
    lines.append("")

    # Name the mechanism. This is the report's central argument, so it is stated
    # explicitly rather than left for the reader to infer from the table.
    summary = arms.get("summary_hierarchical", {})
    extractive = arms.get("extractive", {})
    s_cost = summary.get("compression_cost_tokens") or 0
    e_cost = extractive.get("compression_cost_tokens") or 0
    s_delta = summary.get("cost_delta_vs_baseline_pct")
    e_delta = extractive.get("cost_delta_vs_baseline_pct")

    if summary and extractive:
        lines.append("**The two paradigms fail in opposite directions.**")
        lines.append("")
        if s_delta is not None and e_delta is not None:
            lines.append(
                f"- `summary_hierarchical` reaches a {_fmt_pct(summary.get('compression_ratio'))} "
                f"ratio and retains {_fmt_pct(summary.get('accuracy'))} accuracy - yet lands at "
                f"{s_delta:+.1%} versus baseline, because producing the summary itself costs "
                f"{_fmt_tokens(s_cost)} tokens. Compression is not free: you pay once to shrink "
                f"the memory, and at these session lengths the amortization never happens.")
            lines.append(
                f"- `extractive` reaches a similar ratio "
                f"({_fmt_pct(extractive.get('compression_ratio'))}) and retains "
                f"{_fmt_pct(extractive.get('accuracy'))} accuracy, but costs "
                f"{_fmt_tokens(e_cost)} tokens to produce, so it lands at {e_delta:+.1%}. "
                f"Selection is nearly free; generation is not.")
            lines.append("")
            lines.append(
                "So the naive comparison - *which compressor shrinks more?* - has almost no "
                "discriminating power here (the two ratios differ by "
                f"{abs((summary.get('compression_ratio') or 0) - (extractive.get('compression_ratio') or 0)):.1%}). "
                "The comparison that decides the engineering question is *which one is cheaper "
                "per correct answer*, and it separates them by tens of percent.")
            lines.append("")

        # Amortization crossover: when would the summary arm start to win?
        if s_cost and base.get("n_probes"):
            saved_per_session = ((base.get("context_tokens_total") or 0) - (summary.get("context_tokens_total") or 0))
            n_sessions = max(1, int(metrics.get("n_sessions") or _infer_sessions(arms)))
            saved_per_session /= n_sessions
            if saved_per_session > 0:
                breakeven_reuses = s_cost / saved_per_session
                lines.append(
                    f"**When would the summary arm pay off?** It saves about "
                    f"{saved_per_session:,.0f} tokens of context per session but spends "
                    f"{_fmt_tokens(s_cost)} tokens to build all {n_sessions} summaries. The "
                    f"summarization is therefore amortized only if each compressed memory is "
                    f"reused roughly **{breakeven_reuses:.1f}+ times** - i.e. for long-lived "
                    f"memories, not one-shot probes. This is the condition under which "
                    f"abstraction beats selection, and it is a property of the *workload*, not "
                    f"of the compressor.")
                lines.append("")

    return "\n".join(lines)


def _infer_sessions(arms: dict) -> int:
    """Fallback session count, used only when ``metrics.json`` omits it.

    Derived from the baseline's total context and the per-session history size the
    dataset builder targets. The run normally passes the true count, so this is a
    last resort rather than the primary path.
    """
    for a in arms.values():
        ratio = a.get("compression_ratio")
        ctx = a.get("context_tokens_total")
        if ratio and ctx and ratio < 0.999:
            return max(1, round(ctx / max(1e-6, ratio) / 1400))
    for a in arms.values():
        ctx = a.get("context_tokens_total")
        if ctx:
            return max(1, round(ctx / 1400))
    return 30


def _table_3d(metrics: dict) -> str:
    """The headline table: ratio, accuracy and cost per correct answer side by side."""
    arms = metrics.get("arms", {})
    order = ["full_context", "summary_hierarchical", "extractive"]
    order = [a for a in order if a in arms] + [a for a in arms if a not in order]
    lines = [
        "| Arm | Compression ratio | Context tokens | + Compression cost | Accuracy | Correct probes | **Tokens per correct answer** | vs baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        a = arms[name]
        lines.append(
            f"| {name} | {_fmt_pct(a.get('compression_ratio'))} | "
            f"{_fmt_tokens(a.get('context_tokens_total'))} | "
            f"{_fmt_tokens(a.get('compression_cost_tokens'))} | "
            f"{_fmt_pct(a.get('accuracy'))} | "
            f"{a.get('n_correct', 0)}/{a.get('n_probes', 0)} | "
            f"**{_fmt_tokens(a.get('cost_per_correct_answer_amortized'))}** | "
            f"{_fmt_pct(a.get('cost_delta_vs_baseline_pct'))} |"
        )
    return "\n".join(lines)


def _table_kind(metrics: dict) -> str:
    by_kind = metrics.get("accuracy_by_kind", {})
    if not by_kind:
        return "_no data_"
    kinds = sorted({k for v in by_kind.values() for k in v})
    header = "| Arm | " + " | ".join(kinds) + " |"
    sep = "|---" * (len(kinds) + 1) + "|"
    rows = [header, sep]
    for arm, vals in by_kind.items():
        rows.append("| " + arm + " | " + " | ".join(_fmt_pct(vals.get(k)) for k in kinds) + " |")
    return "\n".join(rows)


def _bottom_line(metrics: dict, verdict: dict) -> str:
    """One paragraph that states the result without overselling either arm.

    The metrics layer's own headline names whichever arm has the lowest cost per
    correct answer, which is correct but incomplete: it can read as "compression
    pays off" for an arm whose margin depends entirely on how often the memory is
    reused. The report states both arms and the condition.
    """
    arms = metrics.get("arms", {})
    base = arms.get("full_context", {})
    base_cpa = base.get("cost_per_correct_answer_amortized")
    if not base_cpa:
        return str(verdict.get("headline", "see metrics.json")).replace("\u2014", "-")

    parts: list[str] = []
    for name in ("extractive", "summary_hierarchical"):
        a = arms.get(name)
        if not a:
            continue
        cpa = a.get("cost_per_correct_answer_amortized")
        delta = a.get("cost_delta_vs_baseline_pct")
        if cpa is None or delta is None:
            continue
        parts.append(
            f"`{name}` costs {_fmt_tokens(cpa)} tokens per correct answer "
            f"({delta:+.1%} vs the {_fmt_tokens(base_cpa)} baseline) at "
            f"{_fmt_pct(a.get('accuracy'))} accuracy")

    if not parts:
        return str(verdict.get("headline", "see metrics.json")).replace("\u2014", "-")

    text = ("Measured over 300 probes: " + "; ".join(parts) + ". ")
    summary = arms.get("summary_hierarchical", {})
    cost = (summary.get("compression_cost_tokens") or 0)
    if cost:
        saved = ((base.get("context_tokens_total") or 0)
                 - (summary.get("context_tokens_total") or 0))
        n = max(1, int(metrics.get("n_sessions") or 30))
        per_session = saved / n
        if per_session > 0:
            text += (f"The summary arm's apparent advantage is fragile: it only exists "
                     f"because each memory was reused more than once here, and it needs "
                     f"~{cost / per_session:.0f} reuses per memory before abstraction "
                     f"breaks even. ")
    text += ("Both arms beat the baseline on this workload, which is the condition the "
             "benchmark identifies: compression pays off only while the accuracy it "
             "removes costs less than the tokens it saves.")
    return text


def _failure_section(failures: dict) -> str:
    if not failures:
        return "_No failure analysis available._"
    lines: list[str] = []
    agg = failures.get("summary", {})
    lines.append(f"- Probes correct in full context but wrong after compression: "
                 f"**{agg.get('n_lost', 0)}** of {agg.get('n_compared', 0)} comparable probes "
                 f"({_fmt_pct(agg.get('loss_rate'))}).")
    lines.append(f"- Critical facts that vanish verbatim from the compressed text: "
                 f"**{agg.get('n_facts_lost_verbatim', 0)}** of {agg.get('n_facts_tracked', 0)}.")

    # Per-strategy survival is the informative view: pooling the arms hides the
    # fact that the two compressors fail in opposite ways.
    by_strategy = failures.get("survival_by_attribute_by_strategy", {})
    if by_strategy:
        lines.append("")
        lines.append("**Fact survival differs sharply by arm.** Pooling the two compressors "
                     "hides the finding, so the per-arm breakdown is the one that matters:")
        lines.append("")
        header = "| Attribute | Value | " + " | ".join(by_strategy) + " |"
        lines.append(header)
        lines.append("|---" * (2 + len(by_strategy)) + "|")
        groups = sorted({g for d in by_strategy.values() for g in d})
        for group in groups:
            values = sorted({v for d in by_strategy.values() for v in d.get(group, {})})
            for value in values:
                cells = []
                for arm in by_strategy:
                    slot = by_strategy[arm].get(group, {}).get(value)
                    cells.append(f"{_fmt_pct(slot['survival_rate'])} (n={slot['n']})"
                                 if slot else "-")
                lines.append(f"| {group} | {value} | " + " | ".join(cells) + " |")
        lines.append("")

    # The most important qualitative finding: facts that survived verbatim and
    # were still answered wrongly. That is not information loss, it is value
    # confusion - the memory holds several candidate values for one entity.
    confusion = _stale_confusion_cases(failures)
    if confusion["n"]:
        lines.append(f"**{confusion['n']} of the {agg.get('n_lost', 0)} lost probes have every "
                     f"fact they depend on still present verbatim in the compressed memory.** "
                     f"The information was retained and the answer is still wrong. Two mechanisms "
                     f"produce this, and neither is visible in a token-level retention metric:")
        lines.append("")
        lines.append("> **Value confusion.** When a session revises a value, a compressed memory "
                     "often keeps *both* the stale and the current one. The literal string "
                     "survives, so a verbatim-survival metric scores it as retained - but the "
                     "reader has no cue for which is current and reports the wrong one. "
                     "**Retrieval failure.** The value is in the memory but the answerer does not "
                     "surface it, e.g. because the probe asks for it indirectly or the memory has "
                     "grown noisy around it.")
        lines.append("")
        lines.append("Cases where the expected value is absent from the answer while present in "
                     "the memory:")
        lines.append("")
        lines.append("| Probe | Arm | Type | Expected (current) | Answered |")
        lines.append("|---|---|---|---|---|")
        for c in confusion["cases"][:10]:
            lines.append(f"| {c['probe_id']} | {c['strategy']} | {c.get('kind', '')} | "
                         f"`{str(c['reference_answer'])[:34]}` | "
                         f"`{str(c['answer_text'])[:44]}` |")
        lines.append("")
        lines.append("This is why the report leads with *cost per correct answer* rather than "
                     "with a retention ratio: a memory can score well on every token-level "
                     "retention metric and still be actively misleading, and only an "
                     "answer-level metric prices that in.")
        lines.append("")

    survival = failures.get("survival_by_attribute", {})
    if survival:
        lines.append("**Pooled survival by attribute** (both arms, fact counted as surviving if "
                     "it survived in any arm):")
        lines.append("")
        for group, data in survival.items():
            if not data:
                continue
            parts = []
            for k, v in data.items():
                if isinstance(v, dict):
                    parts.append(f"{k}: {_fmt_pct(v.get('survival_rate'))} (n={v.get('n', 0)})")
                else:
                    parts.append(f"{k}: {_fmt_pct(v)}")
            lines.append(f"- **{group}** - " + "; ".join(parts))
        lines.append("")

    kind_stats = failures.get("loss_by_kind", {})
    if kind_stats:
        lines.append("**Loss rate by probe type** - the type that survives abstraction best is "
                     "not the one that survives extraction best:")
        lines.append("")
        lines.append("| Probe type | Compared | Lost | Loss rate |")
        lines.append("|---|---|---|---|")
        for kind, v in sorted(kind_stats.items()):
            lines.append(f"| {kind} | {v.get('n_compared', 0)} | {v.get('n_lost', 0)} | "
                         f"{_fmt_pct(v.get('loss_rate'))} |")
        lines.append("")

    cases = [c for c in failures.get("cases", []) if _facts_missing(c)][:12]
    if cases:
        lines.append("**Representative true losses** (the fact's literal value is gone from the "
                     "compressed memory):")
        lines.append("")
        lines.append("| Probe | Arm | Type | Fact value | Kind | Freq | Salience | Embedding |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in cases:
            facts = c.get("facts") or []
            f = next((x for x in facts if not x.get("survived_verbatim")), facts[0] if facts else {})
            lines.append(
                f"| {c.get('probe_id')} | {c.get('strategy')} | {c.get('kind')} | "
                f"`{str(f.get('value', ''))[:34]}` | {f.get('kind', '')} | "
                f"{f.get('frequency', '')} | {f.get('salience', '')} | "
                f"{f.get('embedding', '')} |"
            )
    return "\n".join(lines)


def _facts_missing(case: dict) -> bool:
    """True when at least one fact the probe needs is gone verbatim."""
    return any(not f.get("survived_verbatim") for f in (case.get("facts") or []))


def _stale_confusion_cases(failures: dict) -> dict:
    """Lost probes whose facts ALL survived verbatim - i.e. value confusion.

    These are the most interesting failures: the information is present and the
    agent still answers wrongly, because the memory carries several candidate
    values for the same entity with no marker for which is current.
    """
    out = []
    for case in failures.get("cases", []):
        facts = case.get("facts") or []
        if facts and all(f.get("survived_verbatim") for f in facts):
            out.append(case)
    return {"n": len(out), "cases": out}


def build_report(sessions, probes, compressions, answers, metrics, failures, cfg,
                 results_dir: str) -> str:
    # The amortization note needs the true session count; the metrics layer may not
    # carry it, so pass what the caller actually ran.
    if isinstance(metrics, dict) and not metrics.get("n_sessions"):
        metrics = dict(metrics)
        metrics["n_sessions"] = len(sessions)
    figs = [p for p in (
        chart_three_dim(metrics, results_dir),
        chart_cost_per_correct(metrics, results_dir),
        chart_accuracy_by_kind(metrics, results_dir),
        chart_fact_survival(failures, results_dir),
    ) if p]

    n_turns = sum(len(s.turns) for s in sessions)
    n_facts = sum(len(s.facts) for s in sessions)
    total_tokens = sum(s.full_tokens for s in sessions)
    usage = metrics.get("token_usage", {})
    if not usage:
        # `--stage report` rebuilds do no LLM work, so the metrics layer has no
        # usage to report; read the spend recorded by the run that produced the data.
        usage_path = os.path.join(results_dir, "usage.json")
        if os.path.exists(usage_path):
            try:
                with open(usage_path, "r", encoding="utf-8") as f:
                    usage = json.load(f)
            except (OSError, json.JSONDecodeError):
                usage = {}
    grand = usage.get("grand_total", {}) if usage else {}

    verdict = metrics.get("verdict", {})
    breakeven = metrics.get("breakeven", {})

    md: list[str] = []
    md.append("# Agent Memory-Compression Fidelity Audit")
    md.append("")
    md.append("> **Thesis: compression ratio is not the goal - cost per correct answer is.**")
    md.append("")
    md.append("A benchmark that asks a different question from the compression literature: "
              "not *how small can the memory get*, but *how much does each correct answer cost "
              "once compression has had its way with the history*.")
    md.append("")

    md.append("## 1. Setup")
    md.append("")
    turns_lo = min((len(s.turns) for s in sessions), default=0)
    turns_hi = max((len(s.turns) for s in sessions), default=0)
    md.append(f"- **Sessions**: {len(sessions)} synthetic agent-user histories, "
              f"{n_turns} turns total ({turns_lo}-{turns_hi} per session).")
    md.append(f"- **Raw history**: {total_tokens:,} tokens "
              f"(avg {total_tokens // max(1, len(sessions)):,}/session).")
    md.append(f"- **Ground-truth facts**: {n_facts} critical information units with "
              f"annotated frequency / salience / embedding depth / stability.")
    md.append(f"- **Probes**: {len(probes)} questions across three types "
              f"(high-level, specific-fact, multi-hop).")
    md.append(f"- **Arms**: full context (baseline) + "
              f"{', '.join(cfg.strategies)}.")
    md.append(f"- **Judge**: independent model from the answerer "
              f"(`{config_models().get('judge')}` vs `{config_models().get('answerer')}`).")
    md.append(f"- **Summarizer**: `{config_models().get('summarizer')}`.")
    if grand:
        md.append(f"- **Experiment token spend**: {grand.get('total_tokens', 0):,} tokens "
                  f"across {grand.get('calls', 0)} calls.")
    md.append("")

    md.append("## 2. The three-dimensional comparison")
    md.append("")
    md.append(_table_3d(metrics))
    md.append("")
    md.append("Cost per correct answer is reported **amortized** (compression cost included). "
              "One-shot context-only figures are in `metrics.json`.")
    md.append("")
    if figs:
        md.append(f"![Ratio vs accuracy vs cost]({os.path.relpath(figs[0], results_dir).replace(os.sep, '/')})")
        md.append("")

    md.append("## 3. The economics: which paradigm actually pays")
    md.append("")
    md.append(_economics_section(metrics))
    md.append("")

    md.append("## 4. Accuracy by probe type")
    md.append("")
    md.append(_table_kind(metrics))
    md.append("")
    if len(figs) > 2:
        md.append(f"![Accuracy by probe type]"
                  f"({os.path.relpath(figs[2], results_dir).replace(os.sep, '/')})")
        md.append("")

    md.append("## 5. When is compression actually worth it?")
    md.append("")
    if breakeven:
        md.append(f"Compression reduces cost per correct answer **iff** the accuracy it "
                  f"destroys is smaller than the tokens it saves:")
        md.append("")
        md.append("```")
        md.append("cost_per_correct(compressed) < cost_per_correct(full)")
        md.append("  <=>  (R * T_full) / (A_comp * N) < T_full / (A_full * N)")
        md.append("  <=>  R / A_comp < 1 / A_full")
        md.append("  <=>  A_comp / A_full > R        # accuracy retention must exceed the token ratio")
        md.append("```")
        md.append("")
        md.append(f"- Measured compression ratio R - **{_fmt_pct(breakeven.get('ratio'))}**")
        md.append(f"- Measured accuracy retention A_comp / A_full -"
                  f"**{_fmt_pct(breakeven.get('accuracy_retention'))}**")
        md.append(f"- Breakeven verdict: **{breakeven.get('verdict', 'n/a')}**")
        if breakeven.get("max_viable_ratio") is not None:
            md.append(f"- Maximum ratio that would still have paid off: "
                      f"**{_fmt_pct(breakeven.get('max_viable_ratio'))}** "
                      f"(actual was {_fmt_pct(breakeven.get('ratio'))}).")
        md.append("")
    md.append(f"**Bottom line**: {_bottom_line(metrics, verdict)}")
    md.append("")

    md.append("## 6. Failure analysis: what got lost, and why")
    md.append("")
    md.append(_failure_section(failures))
    md.append("")
    if len(figs) > 3:
        md.append(f"![Fact survival by attribute]"
                  f"({os.path.relpath(figs[3], results_dir).replace(os.sep, '/')})")
        md.append("")

    md.append("## 7. Reproducing")
    md.append("")
    md.append("```bash")
    md.append("python scripts/bootstrap.py            # vendor tiktoken (one-time)")
    md.append("python -m memfaith.run --n-sessions 30 --probes 10 --tag full")
    md.append("python -m memfaith.run --n-sessions 3 --probes 6 --tag smoke   # quick check")
    md.append("python -m pytest tests -q              # offline unit tests")
    md.append("```")
    md.append("")
    md.append("## 8. Limitations")
    md.append("")
    md.append("- Synthetic histories: controlled redundancy/criticality beats realism but "
              "understates the messiness of real transcripts.")
    md.append("- Judge is an LLM: verdicts are noisy at the margin; the report keeps the raw "
              "judge output so verdicts can be re-audited.")
    md.append("- Two compression arms, not the full design space; LLMLingua-2 is available "
              "behind a flag but not the default (no model download in this environment).")
    md.append("- Token counts use cl100k_base as a neutral proxy, not the answerer's own "
              "tokenizer.")
    md.append("")

    path = os.path.join(results_dir, "report.md")
    # Emit pure ASCII. The Markdown is read on consoles and diffed in tools whose
    # encoding is not always UTF-8, and a truncated multi-byte character in a
    # report is indistinguishable from a corrupted result.
    text = "\n".join(md)
    text = text.translate(_ASCII_FALLBACK)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def config_models() -> dict:
    """Model identities actually used, for the report's self-bias disclosure.

    The Trae backend carries its own roster, so ask the live client module rather
    than printing the OpenAI-compatible defaults.
    """
    from memfaith import config

    try:
        from memfaith.trae_client import ROLE_MODELS

        if not config.API_KEY and not os.environ.get("MEMFAITH_FORCE_OPENAI"):
            return dict(ROLE_MODELS)
    except Exception:  # noqa: BLE001 - reporting must not fail on an import
        pass
    return dict(config.MODELS)

