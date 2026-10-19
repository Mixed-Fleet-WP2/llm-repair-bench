"""
token_cost_report.py

Token cost of trace vs no-trace repair, from the per-attempt `usage` blocks
the pipeline records (llm_fixer.LAST_CALL). Reads a probe run made with both
benchmarks into <probe>/results_no_trace and <probe>/results_trace and writes
<probe>/token_cost_report.json.

Per model and condition: the model the provider says answered, outcome on the
mission's original properties, attempts, and summed tokens over all attempts
(input, output, reasoning — the hidden part of output — visible output, cached
input). Then the trace − no-trace difference per model and in total.

Token fields a provider does not report are null (e.g. Anthropic and Ollama
report no separate reasoning tokens; Anthropic counts any thinking in output).

Usage (from the repo root):
    python goal_reachability/token_cost_report.py
    python goal_reachability/token_cost_report.py --probe goal_reachability/token_probe
"""

import argparse
import json
from pathlib import Path

HERE     = Path(__file__).resolve().parent
GOALS    = HERE.parent / "dataset"
TOKEN_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")


def original_queries(mission: str) -> list[str]:
    """The mission's original (repair-target) properties: its queries.q minus
    the goal-reachability queries added by make_goal_dataset.py."""
    d = GOALS / mission
    goals = {l.strip() for l in (d / "goal_queries.q").read_text().splitlines() if l.strip()}
    return [l.strip() for l in (d / "queries.q").read_text().splitlines()
            if l.strip() and l.strip() not in goals]


def summarize_run(results_file: Path) -> dict:
    missions = json.loads(results_file.read_text())
    attempts = [a for m in missions for a in m["attempts"]]
    usages = [a.get("usage") or {} for a in attempts]

    tokens = {}
    for k in TOKEN_KEYS:
        vals = [u[k] for u in usages if k in u]
        tokens[k] = sum(vals) if vals else None
    if tokens["output_tokens"] is not None:
        tokens["visible_output_tokens"] = tokens["output_tokens"] - (tokens["reasoning_tokens"] or 0)
    else:
        tokens["visible_output_tokens"] = None
    tokens["total_tokens"] = (None if tokens["input_tokens"] is None
                              else tokens["input_tokens"] + (tokens["output_tokens"] or 0))

    orig_total = sum(len(original_queries(m["mission"])) for m in missions)
    orig_passed = sum(q in m["final_passing"] for m in missions
                      for q in original_queries(m["mission"]))
    return {
        "missions": [m["mission"] for m in missions],
        "served_models": sorted({u["served_model"] for u in usages if u.get("served_model")}),
        "status": [m["status"] for m in missions],
        "original_properties_repaired": f"{orig_passed}/{orig_total}",
        "attempts": len(attempts),
        "goal_rejected_attempts": sum(any(i > 2 for i in a["regressions"]) for a in attempts),
        "invalid_attempts": sum(not a["llm_valid"] for a in attempts),
        "attempts_without_usage": sum(not u for u in usages),
        "tokens": tokens,
        # prompt size independent of how many attempts were needed
        "input_tokens_per_attempt": (round(tokens["input_tokens"] / len(attempts))
                                     if tokens["input_tokens"] is not None and attempts else None),
        "elapsed_s": round(sum(m["elapsed_s"] for m in missions), 1),
    }


def diff(t: dict, n: dict) -> dict:
    out = {}
    for k in (*TOKEN_KEYS, "visible_output_tokens", "total_tokens"):
        a, b = t["tokens"].get(k), n["tokens"].get(k)
        if a is None or b is None:
            out[k] = None
            continue
        out[k] = {"absolute": a - b, "relative_pct": round(100 * (a - b) / b, 1) if b else None}
    out["attempts"] = t["attempts"] - n["attempts"]
    a, b = t.get("input_tokens_per_attempt"), n.get("input_tokens_per_attempt")
    out["input_tokens_per_attempt"] = (None if a is None or b is None else
                                       {"absolute": a - b, "relative_pct": round(100 * (a - b) / b, 1)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Token cost report, trace vs no trace.")
    ap.add_argument("--probe", default=str(HERE / "token_probe"))
    args = ap.parse_args()
    probe = Path(args.probe)

    runs = {cond: {f.name[: -len("_results.json")]: summarize_run(f)
                   for f in sorted((probe / f"results_{cond}").glob("*_results.json"))}
            for cond in ("no_trace", "trace")}
    models = sorted(set(runs["no_trace"]) | set(runs["trace"]))

    per_model, totals = {}, {"no_trace": {}, "trace": {}}
    for m in models:
        n, t = runs["no_trace"].get(m), runs["trace"].get(m)
        per_model[m] = {"no_trace": n, "trace": t,
                        "trace_minus_no_trace": diff(t, n) if n and t else None}

    for cond in ("no_trace", "trace"):
        for k in (*TOKEN_KEYS, "visible_output_tokens", "total_tokens"):
            vals = [r["tokens"][k] for r in runs[cond].values() if r["tokens"][k] is not None]
            totals[cond][k] = sum(vals) if vals else None
        totals[cond]["attempts"] = sum(r["attempts"] for r in runs[cond].values())
    both = [m for m in models if per_model[m]["trace_minus_no_trace"]]
    totals["trace_minus_no_trace"] = diff(
        {"tokens": totals["trace"], "attempts": totals["trace"]["attempts"]},
        {"tokens": totals["no_trace"], "attempts": totals["no_trace"]["attempts"]})

    report = {
        "description": "Token usage of LLM repair with vs without counterexample traces, "
                       "summed over all attempts, as reported by each provider's API.",
        "dataset": "goal_reachability/dataset",
        "missions": sorted({ms for c in runs.values() for r in c.values() for ms in r["missions"]}),
        "models_in_both_conditions": both,
        "notes": [
            "output_tokens includes reasoning_tokens where the provider reports them; "
            "visible_output_tokens = output_tokens - reasoning_tokens.",
            "null = not reported by that provider (Anthropic and Ollama report no "
            "separate reasoning tokens; Anthropic counts any thinking in output_tokens).",
            "Single run per model and condition; token counts vary between runs, "
            "especially reasoning tokens.",
            "served_models lists the model id the provider reported; the xAI labels "
            "grok-3-mini and grok-code-fast were answered by grok-4.3.",
        ],
        "totals": totals,
        "per_model": per_model,
    }
    out = probe / "token_cost_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'model':<20}{'no-trace tok':>14}{'trace tok':>12}{'Δ total':>10}"
          f"{'Δ in/att':>10}{'att N/T':>9}{'repaired N/T':>14}  served")
    for m in both:
        r = per_model[m]
        n, t = r["no_trace"], r["trace"]
        d = r["trace_minus_no_trace"]
        fmt = lambda v: "n/a" if v is None else f"{v:,}"
        pct = lambda x: "n/a" if not x else f"{x['relative_pct']:+.0f}%"
        print(f"{m:<20}{fmt(n['tokens']['total_tokens']):>14}{fmt(t['tokens']['total_tokens']):>12}"
              f"{pct(d['total_tokens']):>10}{pct(d['input_tokens_per_attempt']):>10}"
              f"{str(n['attempts']) + '/' + str(t['attempts']):>9}"
              f"{n['original_properties_repaired'] + ' / ' + t['original_properties_repaired']:>14}"
              f"  {', '.join(t['served_models'] or n['served_models'])}")
    tt = totals["trace_minus_no_trace"]
    print(f"\nAll models: {totals['no_trace']['total_tokens']:,} → {totals['trace']['total_tokens']:,} "
          f"tokens ({tt['total_tokens']['relative_pct']:+.1f}%); input "
          f"{tt['input_tokens']['relative_pct']:+.1f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
