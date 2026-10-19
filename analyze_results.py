"""
analyze_results.py

Aggregates <dir>/*_results.json into one per-model stats file:
<dir>/analytics.json. Defaults to results/ (the trace-based benchmark.py
output); pass --dir results_no_trace to analyze the no-trace ablation
instead.

Usage:
    python analyze_results.py                       # results/
    python analyze_results.py --dir results_no_trace
"""

import argparse
import json
from pathlib import Path


def analyze_model(path: Path) -> dict:
    model = path.stem.replace("_results", "")
    data = json.loads(path.read_text())
    n = len(data)

    fixed   = sum(1 for d in data if d["status"] == "FIXED")
    partial = sum(1 for d in data if d["status"] == "PARTIAL")
    failed  = sum(1 for d in data if d["status"] == "FAILED")
    error   = sum(1 for d in data if d["status"] == "ERROR")
    skip    = sum(1 for d in data if d["status"] == "SKIP")

    considered   = [d for d in data if d["status"] in ("FIXED", "PARTIAL", "FAILED")]
    avg_attempts = sum(d["attempts_used"] for d in considered) / max(1, len(considered))
    avg_time_s   = sum(d["elapsed_s"] for d in data) / max(1, n)
    total_time_s = sum(d["elapsed_s"] for d in data)

    total_regressions = sum(
        1 for d in data for a in d.get("attempts", []) if a.get("regressions")
    )
    invalid_xml_attempts = sum(
        1 for d in data for a in d.get("attempts", []) if not a.get("llm_valid", True)
    )

    # Token usage and served model, recorded per attempt since usage logging
    # was added (older runs have no "usage" and report None here).
    usages = [a["usage"] for d in data for a in d.get("attempts", []) if a.get("usage")]
    def token_sum(key):
        vals = [u[key] for u in usages if key in u]
        return sum(vals) if vals else None
    served_models = sorted({u["served_model"] for u in usages if u.get("served_model")})

    total_queries        = sum(d["n_queries"] for d in data)
    total_final_passing  = sum(len(d["final_passing"]) for d in data)

    return {
        "model":                 model,
        "n_missions":            n,
        "fixed":                 fixed,
        "partial":               partial,
        "failed":                failed,
        "error":                 error,
        "skip":                  skip,
        "fix_rate":              round(fixed / n * 100, 2) if n else 0.0,
        "query_pass_rate":       round(total_final_passing / total_queries * 100, 2) if total_queries else 0.0,
        "avg_attempts":          round(avg_attempts, 2),
        "avg_time_s":            round(avg_time_s, 2),
        "total_time_s":          round(total_time_s, 2),
        "total_regressions":     total_regressions,
        "invalid_xml_attempts":  invalid_xml_attempts,
        "attempts_with_usage":   len(usages),
        "input_tokens":          token_sum("input_tokens"),
        "output_tokens":         token_sum("output_tokens"),
        "reasoning_tokens":      token_sum("reasoning_tokens"),
        "cached_input_tokens":   token_sum("cached_input_tokens"),
        "served_models":         served_models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results",
                         help="Results folder to analyze (default: results)")
    args = parser.parse_args()

    results_dir    = Path(__file__).parent / args.dir
    analytics_file = results_dir / "analytics.json"

    model_files = sorted(results_dir.glob("*_results.json"))
    if not model_files:
        print(f"No *_results.json files found in {results_dir}/")
        return

    stats = [analyze_model(f) for f in model_files]
    stats.sort(key=lambda s: -s["fix_rate"])

    analytics_file.write_text(json.dumps(stats, indent=2))
    print(f"Wrote per-model analytics for {len(stats)} model(s) → {analytics_file}")
    for s in stats:
        print(f"  {s['model']:<22} fix_rate={s['fix_rate']:5.1f}%  "
              f"query_pass_rate={s['query_pass_rate']:5.1f}%  "
              f"avg_attempts={s['avg_attempts']:.2f}  avg_time_s={s['avg_time_s']:.1f}")


if __name__ == "__main__":
    main()
