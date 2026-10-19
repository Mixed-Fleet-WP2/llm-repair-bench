"""
make_figures.py

Aggregates <dir>/*_results.json across all benchmarked models and renders
publication-ready static figures (PDF + PNG, 300dpi) into <out>/:

  fig1_fix_rate_by_model   - stacked horizontal bar: Fixed/Partial/Failed per model
  fig2_speed_vs_accuracy   - scatter: avg time per attempt (log) vs full-fix rate
  fig3_hardest_missions    - horizontal bar: hardest missions by model failure rate

Also writes <out>/benchmark_summary.csv with the underlying per-model numbers.

Two models are excluded from the with-trace ranking (gemini-2.0-flash,
deepseek-r1:32b): every call to them returned HTTP 404 (deprecated endpoint /
missing local Ollama tag), so their 0% reflects infra breakage, not repair
ability. Harmless no-op if run against a directory that doesn't have them.

Usage:
    python make_figures.py                                     # results/ -> figures/
    python make_figures.py --dir results_no_trace --out figures_no_trace
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from adjustText import adjust_text

ROOT = Path(__file__).parent

EXCLUDED = {
    "gemini-2.0-flash": "API endpoint deprecated (HTTP 404 on every call)",
    "deepseek-r1-32b":  "Ollama model tag not found locally (HTTP 404 on every call)",
}

# Colors from the project's validated status palette.
COLOR_FIXED   = "#0ca30c"
COLOR_PARTIAL = "#fab219"
COLOR_FAILED  = "#ec835a"
COLOR_SERIES  = "#2a78d6"
COLOR_MUTED   = "#898781"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_results(results_dir):
    summary = []
    mission_stats = {}

    for f in sorted(results_dir.glob("*_results.json")):
        model = f.stem.replace("_results", "")
        data = json.loads(f.read_text())
        n = len(data)
        fixed = sum(1 for d in data if d["status"] == "FIXED")
        partial = sum(1 for d in data if d["status"] == "PARTIAL")
        failed = sum(1 for d in data if d["status"] == "FAILED")

        considered = [d for d in data if d["status"] in ("FIXED", "PARTIAL", "FAILED")]
        avg_attempts = sum(d["attempts_used"] for d in considered) / max(1, len(considered))
        avg_time = sum(d["elapsed_s"] for d in data) / max(1, n)

        invalid_xml = sum(
            1 for d in data for a in d.get("attempts", []) if not a.get("llm_valid", True)
        )
        total_regressions = sum(
            1 for d in data for a in d.get("attempts", []) if a.get("regressions")
        )
        total_queries = sum(d["n_queries"] for d in data)
        total_final_passing = sum(len(d["final_passing"]) for d in data)

        summary.append({
            "model": model,
            "excluded": model in EXCLUDED,
            "exclude_reason": EXCLUDED.get(model, ""),
            "n_missions": n,
            "fixed": fixed,
            "partial": partial,
            "failed": failed,
            "fix_rate": fixed / n * 100,
            "query_pass_rate": total_final_passing / max(1, total_queries) * 100,
            "avg_attempts": avg_attempts,
            "avg_time_s": avg_time,
            "invalid_xml_attempts": invalid_xml,
            "total_regressions": total_regressions,
        })

        if model not in EXCLUDED:
            for d in data:
                ms = mission_stats.setdefault(d["mission"], {"fixed": 0, "not_fixed": 0})
                if d["status"] == "FIXED":
                    ms["fixed"] += 1
                else:
                    ms["not_fixed"] += 1

    valid = [s for s in summary if not s["excluded"]]
    excluded = [s for s in summary if s["excluded"]]
    valid.sort(key=lambda x: (-x["fix_rate"], x["avg_time_s"]))

    missions = []
    for name, st in mission_stats.items():
        total = st["fixed"] + st["not_fixed"]
        missions.append({
            "mission": name,
            "fail_rate": st["not_fixed"] / total * 100,
            "total_models": total,
        })
    missions.sort(key=lambda x: -x["fail_rate"])

    return valid, excluded, missions


def fig1_fix_rate_by_model(valid, excluded, out_base, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(valid) + 1.2))

    models = [m["model"] for m in valid][::-1]
    y = range(len(models))

    fixed = [m["fixed"] for m in valid][::-1]
    partial = [m["partial"] for m in valid][::-1]
    failed = [m["failed"] for m in valid][::-1]

    ax.barh(y, fixed, color=COLOR_FIXED, label="Fixed", height=0.62)
    ax.barh(y, partial, left=fixed, color=COLOR_PARTIAL, label="Partial", height=0.62)
    left2 = [f + p for f, p in zip(fixed, partial)]
    ax.barh(y, failed, left=left2, color=COLOR_FAILED, label="Failed", height=0.62)

    for i, m in enumerate(valid[::-1]):
        ax.text(m["n_missions"] + 0.4, i, f"{m['fix_rate']:.0f}%",
                 va="center", ha="left", fontsize=9, color="#333")

    ax.set_yticks(list(y))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Missions (of 30)")
    ax.set_xlim(0, 33)
    ax.set_title(f"Repair outcome by model (30 missions, ≤3 attempts each){title_suffix}", fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)

    excl_names = ", ".join(m["model"] for m in excluded)
    if excl_names:
        fig.text(0.01, -0.06, f"Excluded (endpoint unavailable): {excl_names}",
                  fontsize=7.5, color=COLOR_MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_speed_vs_accuracy(valid, out_base, title_suffix=""):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    xs = [m["avg_time_s"] for m in valid]
    ys = [m["fix_rate"] for m in valid]
    ax.scatter(xs, ys, s=42, color=COLOR_SERIES, zorder=3, edgecolors="white", linewidths=1)

    texts = [
        ax.text(x, y, m["model"], fontsize=8, color="#333")
        for x, y, m in zip(xs, ys, valid)
    ]
    adjust_text(
        texts, x=xs, y=ys, ax=ax,
        arrowprops=dict(arrowstyle="-", color=COLOR_MUTED, lw=0.7),
        expand=(1.3, 1.6),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Avg. time per mission attempt, seconds (log scale)")
    ax.set_ylabel("Full-fix rate (%)")
    ax.set_ylim(-3, 108)
    ax.set_title(f"Speed vs. accuracy across models{title_suffix}", fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_hardest_missions(missions, out_base, top_n=15, title_suffix=""):
    top = missions[:top_n][::-1]
    labels = [m["mission"].split("_", 1)[1].replace("_", " ") for m in top]
    rates = [m["fail_rate"] for m in top]
    colors = [COLOR_FAILED if r >= 50 else COLOR_PARTIAL if r >= 25 else COLOR_FIXED for r in rates]

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * top_n + 1))
    y = range(len(top))
    ax.barh(y, rates, color=colors, height=0.62)
    for i, r in enumerate(rates):
        ax.text(r + 1, i, f"{r:.0f}%", va="center", ha="left", fontsize=8.5, color="#333")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Share of evaluated models that failed to fully fix the mission")
    ax.set_xlim(0, 100)
    ax.set_title(f"Hardest missions (top {top_n} of {len(missions)}){title_suffix}", fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(valid, excluded, out_path):
    rows = sorted(valid + excluded, key=lambda m: -m["fix_rate"])
    fields = ["model", "excluded", "exclude_reason", "n_missions", "fixed", "partial",
              "failed", "fix_rate", "query_pass_rate", "avg_attempts", "avg_time_s",
              "invalid_xml_attempts", "total_regressions"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results",
                         help="Results folder to read *_results.json from (default: results)")
    parser.add_argument("--out", default=None,
                         help="Figures output folder (default: 'figures', or 'figures_no_trace' "
                              "if --dir is results_no_trace)")
    args = parser.parse_args()

    results_dir = ROOT / args.dir
    out_dir = ROOT / (args.out if args.out else ("figures_no_trace" if args.dir == "results_no_trace" else "figures"))
    title_suffix = " — no trace" if "no_trace" in args.dir else (" — with trace" if args.dir == "results" else f" — {args.dir}")

    out_dir.mkdir(exist_ok=True)
    valid, excluded, missions = load_results(results_dir)

    fig1_fix_rate_by_model(valid, excluded, out_dir / "fig1_fix_rate_by_model", title_suffix)
    fig2_speed_vs_accuracy(valid, out_dir / "fig2_speed_vs_accuracy", title_suffix)
    fig3_hardest_missions(missions, out_dir / "fig3_hardest_missions", title_suffix=title_suffix)
    write_csv(valid, excluded, out_dir / "benchmark_summary.csv")

    print(f"Wrote figures + summary CSV to {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
