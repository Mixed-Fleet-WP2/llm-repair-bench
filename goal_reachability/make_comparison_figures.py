"""
make_comparison_figures.py

Trace vs no-trace on the goal-reachability dataset, judged by orig_pass_rate:
original properties passing / all original properties (the 60 repair
targets), with and without the counterexample trace. Same number as
orig_pass_rate in evaluation/summary.json.

Counted queries are the 60 repair targets (30 missions x 2 original
properties, all failing before repair), read from each run's
results_{trace,no_trace}/<model>_results.json (final_passing). The 60 goal
queries are left out of the counts: they pass in every run (the harness
rejects any attempt that breaks one), so they would add the same 60 to every
bar.

Writes PDF + PNG (300 dpi) to goal_reachability/figures_comparison/:

  fig_cmp1_queries_passed       orig_pass_rate per model, with vs without trace
  fig_cmp2_queries_delta        queries gained / lost per model with the trace
  fig_cmp3_by_fault_category    queries passed per fault category, all models
                                pooled, with vs without trace
  fig_cmp4_category_heatmap     model x fault category: queries gained / lost
  queries_passed.csv            every count in the figures

Corrections listed in goal_reachability/corrections.json (pipeline defects,
re-verified offline) are applied on read; raw result files are not modified.

Usage (from the repo root):
    python goal_reachability/make_comparison_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch

HERE     = Path(__file__).resolve().parent
GOALS    = HERE.parent / "dataset"
OUT_DIR  = HERE / "figures_comparison"
RUNS     = {"trace": HERE / "results_trace", "no_trace": HERE / "results_no_trace"}
CORRECTIONS = HERE / "corrections.json"   # documented fixes for pipeline defects

# Two conditions of one measurement: one hue, two shades (validated; the light
# shade is below 3:1 contrast, so every bar carries a direct value label).
COLOR_TRACE    = "#1c5cab"
COLOR_NO_TRACE = "#5598e7"
# Polarity (trace helped / hurt): diverging blue <-> red, gray midpoint.
COLOR_BETTER   = "#2a78d6"
COLOR_WORSE    = "#e34948"
TEXT_PRIMARY   = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_PRIMARY,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# LLM names exactly as written in the paper.
DISPLAY_NAME = {
    "deepseek-reasoner": "DeepSeek Reasoner",
    "gemini-3.6-flash":  "Gemini 3.6 Flash",
    "gemini-2.5-pro":    "Gemini 2.5 Pro",
    "gpt-5.1":           "GPT-5.1",
    "gpt-5-mini":        "GPT-5-mini",
    "gpt-5-nano":        "GPT-5-nano",
    "gpt-4o":            "GPT-4o",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-haiku-4-5":  "Claude Haiku 4.5",
    "grok-3-mini":       "Grok 3 Mini",
    "grok-code-fast":    "Grok Code Fast",
    "qwen3.5-9b":        "Qwen3.5-9B",
}


def display(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


CATEGORY_LABEL = {
    "mutual_exclusion":  "Mutual exclusion",
    "ordering":          "Ordering / prerequisite",
    "missing_guard":     "Missing guard",
    "timing":            "Time bound",
    "capacity":          "Capacity",
    "liveness":          "Conditional liveness",
}


def fault_category(raw: str) -> str:
    c = raw.lower()
    if "mutual" in c:                          return "mutual_exclusion"
    if "order" in c or "prereq" in c:          return "ordering"
    if "time" in c:                            return "timing"
    if "capacity" in c:                        return "capacity"
    if "liveness" in c or "response" in c:     return "liveness"
    return "missing_guard"


# ── data ──────────────────────────────────────────────────────────────────────

def target_queries() -> dict[str, dict[str, str]]:
    """mission -> {original query formula -> fault category}."""
    out = {}
    for m in sorted(p for p in GOALS.iterdir() if p.is_dir()):
        goals = {l.strip() for l in (m / "goal_queries.q").read_text().splitlines() if l.strip()}
        qs = [l.strip() for l in (m / "queries.q").read_text().splitlines()
              if l.strip() and l.strip() not in goals]
        faults = json.loads((m / "metadata.json").read_text())["faults"]
        out[m.name] = {q: fault_category(f["category"]) for q, f in zip(qs, faults)}
    return out


def load_corrections() -> dict:
    """(condition, model, mission) -> corrected number of original properties passing."""
    if not CORRECTIONS.exists():
        return {}
    return {(c["condition"], c["model"], c["mission"]): c["corrected"]["original_properties_passing"]
            for c in json.loads(CORRECTIONS.read_text())["corrections"]}


def load():
    targets = target_queries()
    corrections = load_corrections()
    passed = {}   # (condition, model) -> set of (mission, query) repaired
    for cond, rdir in RUNS.items():
        for f in sorted(rdir.glob("*_results.json")):
            model = f.name[: -len("_results.json")]
            got = set()
            for r in json.loads(f.read_text()):
                for q in r["final_passing"]:
                    if q in targets.get(r["mission"], {}):
                        got.add((r["mission"], q))
                if corrections.get((cond, model, r["mission"])) == len(targets[r["mission"]]):
                    # corrected to a full repair: every original property passes
                    got |= {(r["mission"], q) for q in targets[r["mission"]]}
            passed[(cond, model)] = got
    models = sorted({m for c, m in passed if c == "trace"} &
                    {m for c, m in passed if c == "no_trace"})
    all_q = [(m, q) for m, qs in targets.items() for q in qs]
    cat_of = {(m, q): c for m, qs in targets.items() for q, c in qs.items()}

    stats = []
    for model in models:
        t, n = passed[("trace", model)], passed[("no_trace", model)]
        s = {"model": model, "trace": len(t), "no_trace": len(n),
             "gained": len(t - n), "lost": len(n - t)}
        s["delta"] = s["trace"] - s["no_trace"]
        for cat in CATEGORY_LABEL:
            qs = [k for k in all_q if cat_of[k] == cat]
            s[f"{cat}_trace"]    = sum(k in t for k in qs)
            s[f"{cat}_no_trace"] = sum(k in n for k in qs)
        stats.append(s)
    cat_total = {cat: sum(c == cat for c in cat_of.values()) for cat in CATEGORY_LABEL}
    return stats, len(all_q), cat_total


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def condition_legend(ax, loc="lower right"):
    ax.legend(handles=[Patch(color=COLOR_NO_TRACE, label="No-Trace"),
                       Patch(color=COLOR_TRACE, label="Trace-Assisted")],
              loc=loc, frameon=False, fontsize=9)


# ── figures ───────────────────────────────────────────────────────────────────

def fig_queries_passed(stats, total):
    order = sorted(stats, key=lambda s: s["no_trace"])
    fewer = sum(s["delta"] < 0 for s in stats)
    fig, ax = plt.subplots(figsize=(8, 0.52 * len(order) + 1.4))
    h = 0.38
    for y, s in enumerate(order):
        for dy, v, c in ((h / 2 + 0.01, s["no_trace"], COLOR_NO_TRACE),
                         (-h / 2 - 0.01, s["trace"], COLOR_TRACE)):
            pct = 100 * v / total
            ax.barh(y + dy, pct, height=h, color=c)
            ax.text(pct + 0.8, y + dy, f"{pct:.1f}%  ({v}/{total})", va="center",
                    fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(order)), [display(s["model"]) for s in order])
    ax.set_xlim(0, 122)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(f"Property Repair Rate (% of {total} violated properties)")
    ax.set_title(f"Properties repaired per LLM — with traces, {fewer} of {len(stats)} "
                 f"LLMs repair fewer", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    condition_legend(ax)
    save(fig, "fig_cmp1_queries_passed")


def fig_queries_delta(stats, total):
    order = sorted(stats, key=lambda s: (s["delta"], s["gained"]))
    fig, ax = plt.subplots(figsize=(8, 0.46 * len(order) + 1.6))
    for y, s in enumerate(order):
        ax.barh(y, s["gained"], height=0.62, color=COLOR_BETTER)
        ax.barh(y, -s["lost"], height=0.62, color=COLOR_WORSE)
        if s["gained"]:
            ax.text(s["gained"] + 0.25, y, f"+{s['gained']}", va="center", fontsize=8,
                    color=TEXT_SECONDARY)
        if s["lost"]:
            ax.text(-s["lost"] - 0.25, y, f"−{s['lost']}", va="center", ha="right",
                    fontsize=8, color=TEXT_SECONDARY)
    lim = max(max(s["gained"], s["lost"]) for s in stats) + 3
    ax.set_xlim(-lim, lim + 10)
    for y, s in enumerate(order):
        d = s["delta"]
        pp = 100 * d / total
        ax.text(lim + 0.8, y, (f"Net: {d:+d} ({pp:+.1f} pp)" if d else "Net: ±0")
                .replace("-", "\u2212"), va="center",
                fontsize=9,
                color=TEXT_PRIMARY if d else TEXT_SECONDARY, fontweight="bold" if d else None)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_yticks(range(len(order)), [display(s["model"]) for s in order])
    ticks = [t for t in ax.get_xticks() if -lim <= t <= lim]
    ax.set_xticks(ticks, [str(abs(int(t))) for t in ticks])
    ax.set_xlabel("Properties repaired only without traces (left) / only with traces (right)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=COLOR_WORSE, label="Repaired only in No-Trace"),
                       Patch(color=COLOR_BETTER, label="Repaired only in Trace-Assisted")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              frameon=False, fontsize=9)
    save(fig, "fig_cmp2_queries_delta")


def fig_by_category(stats, cat_total):
    n_models = len(stats)
    cats = sorted(CATEGORY_LABEL, key=lambda c: sum(s[f"{c}_no_trace"] for s in stats)
                  / (cat_total[c] * n_models))
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(cats) + 1.4))
    h = 0.38
    # Title names the category losing the most queries (a count, not a %, so a
    # 4-query category cannot dominate on a handful of queries).
    losses = {c: sum(s[f"{c}_no_trace"] - s[f"{c}_trace"] for s in stats) for c in cats}
    worst = [c for c in cats if losses[c] == max(losses.values()) and losses[c] > 0]
    for y, c in enumerate(cats):
        denom = cat_total[c] * n_models
        vals = {k: sum(s[f"{c}_{k}"] for s in stats) for k in ("no_trace", "trace")}
        d = 100 * (vals["trace"] - vals["no_trace"]) / denom
        for dy, k, col in ((h / 2 + 0.01, "no_trace", COLOR_NO_TRACE),
                           (-h / 2 - 0.01, "trace", COLOR_TRACE)):
            pct = 100 * vals[k] / denom
            ax.barh(y + dy, pct, height=h, color=col)
            ax.text(pct + 0.8, y + dy, f"{vals[k]}/{denom}", va="center", fontsize=8,
                    color=TEXT_SECONDARY)
        ax.text(1.03, y, f"{d:+.1f} pp" if round(d, 1) else "±0", va="center", fontsize=9,
                color=TEXT_PRIMARY, transform=ax.get_yaxis_transform())
    ax.text(1.03, len(cats) - 0.4, "Δ trace", fontsize=9, color=TEXT_SECONDARY,
            transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(cats)),
                  [f"{CATEGORY_LABEL[c]}  ({cat_total[c]} properties)" for c in cats])
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(f"Properties repaired, all {n_models} LLMs pooled (%)")
    title = "Properties repaired by fault category, all LLMs pooled"
    if worst:
        names = " and ".join(CATEGORY_LABEL[c].lower() for c in worst)
        title += f"\nTraces cost the most on {names} (−{losses[worst[0]]} properties each)" \
            if len(worst) > 1 else \
            f"\nTraces cost the most on {names} (−{losses[worst[0]]} properties)"
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    condition_legend(ax)
    save(fig, "fig_cmp3_by_fault_category")


def fig_category_heatmap(stats, cat_total):
    order = sorted(stats, key=lambda s: s["delta"])
    cats = list(CATEGORY_LABEL)
    grid = [[s[f"{c}_trace"] - s[f"{c}_no_trace"] for c in cats] for s in order]
    lim = max(1, max(abs(v) for row in grid for v in row))
    cmap = LinearSegmentedColormap.from_list("div", ["#c7302f", "#f0efec", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 2))
    im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    def two_lines(label: str) -> str:
        return label.replace(" / ", " /\n") if " / " in label else label.replace(" ", "\n", 1)
    ax.set_xticks(range(len(cats)),
                  [f"{two_lines(CATEGORY_LABEL[c])}\n({cat_total[c]} properties)" for c in cats],
                  fontsize=8)
    ax.set_yticks(range(len(order)), [display(s["model"]) for s in order])
    ax.set_xticks([x - 0.5 for x in range(1, len(cats))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            ax.text(x, y, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > lim * 0.6 else
                    (TEXT_PRIMARY if v else TEXT_SECONDARY))
    ax.set_title("Properties gained (+) / lost (−) with traces, per LLM and fault category",
                 loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("properties (Trace-Assisted − No-Trace)", color=TEXT_SECONDARY)
    cb.outline.set_visible(False)
    save(fig, "fig_cmp4_category_heatmap")


def write_table(stats, total, cat_total):
    fields = (["model", "queries_total", "passed_no_trace", "passed_trace", "delta",
               "gained_with_trace", "lost_with_trace"] +
              [f"{c}_{k}" for c in CATEGORY_LABEL for k in ("no_trace", "trace")])
    with open(OUT_DIR / "queries_passed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in sorted(stats, key=lambda s: -s["no_trace"]):
            w.writerow([s["model"], total, s["no_trace"], s["trace"], s["delta"],
                        s["gained"], s["lost"]] +
                       [s[f"{c}_{k}"] for c in CATEGORY_LABEL for k in ("no_trace", "trace")])
        w.writerow(["(queries per category)", total, "", "", "", "", ""] +
                   [cat_total[c] for c in CATEGORY_LABEL for _ in (0, 1)])
    print("  queries_passed.csv")


def main():
    stats, total, cat_total = load()
    if not stats:
        raise SystemExit("No model has results in both conditions yet — run both "
                         "run_trace.sh and run_no_trace.sh first.")
    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*"):          # figures from earlier versions of this script
        if old.is_file():
            old.unlink()
    print(f"{len(stats)} models in both conditions → {OUT_DIR}")
    fig_queries_passed(stats, total)
    fig_queries_delta(stats, total)
    fig_by_category(stats, cat_total)
    fig_category_heatmap(stats, cat_total)
    write_table(stats, total, cat_total)


if __name__ == "__main__":
    main()
