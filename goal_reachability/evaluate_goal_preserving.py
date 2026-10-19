"""
evaluate_goal_preserving.py

Fair, run-independent scoring of repair results. Goal-reachability queries are
used only as a GATE, never as credit: they already pass on the faulty model,
so counting them as passed queries would inflate every score.

For each model x mission, the final model (repaired.xml, or faulty.xml when
the run produced no repair) is re-verified with verifyta against

  - the ORIGINAL properties   (queries.q minus goal_queries.q — all fail before repair)
  - the GOAL queries          (dataset/<m>/goal_queries.q)

The gate uses only the per-robot goals (`E<> robot.Goal`). The joint goal
`E<> (a.Goal && b.Goal)` is reported separately and does not gate: it also
fails for correct repairs that add a location after a robot's goal (the robot
finishes, then moves on, so it never sits in the old sink while the other one
does) — e.g. deepseek-reasoner on missions 05 and 09.

Metrics per model:

  orig_pass_rate      original properties passing / all original properties
                      (ungated — what the original benchmark measured)
  goal_preserved      missions whose final model keeps every per-robot goal
                      reachable
  joint_goal_kept     missions whose final model also passes the joint goal
                      (diagnostic only)
  gated_pass_rate     original properties passing, counted only in missions
                      where the goal is preserved  <- the fair headline metric
  genuine_fix_rate    missions with all original properties passing AND the
                      goal preserved
  vacuous_fixes       missions with all original properties passing but a goal
                      broken (counted as FIXED by the original benchmark)

By default only the runs inside goal_reachability/ are scored (every
goal_reachability/results_* folder), written to goal_reachability/evaluation/.
The original runs in the repo root are scored only on request
(--include-original), into a separate goal_reachability/evaluation/vs_original/
so the two experiments never mix. Because every run is re-verified the same
way, the numbers in vs_original/ are directly comparable.

Usage (from the repo root):
    python goal_reachability/evaluate_goal_preserving.py
    python goal_reachability/evaluate_goal_preserving.py --include-original
"""

import argparse
import concurrent.futures as cf
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from llm_fixer import VERIFYTA  # noqa: E402

GOALS    = ROOT / "dataset"
OUT      = HERE / "evaluation"
_CORR    = HERE / "corrections.json"
CORRECTIONS = ({(c["condition"], c["model"], c["mission"]): c
                for c in json.loads(_CORR.read_text())["corrections"]}
               if _CORR.exists() else {})

ORIGINAL_RUNS = {
    "original_trace":    ROOT / "results",
    "original_no_trace": ROOT / "results_no_trace",
}


def own_runs() -> dict[str, Path]:
    """Every results_* folder directly inside goal_reachability/."""
    return {"goal_reachability_" + p.name.removeprefix("results_"): p
            for p in sorted(HERE.glob("results_*")) if p.is_dir()}
# Models whose runs were pure infrastructure failure (HTTP 404 on every call).
EXCLUDED = {"deepseek-r1-32b", "gemini-2.0-flash"}


def read_queries(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def verify(xml: Path, queries: list[str]) -> list[bool] | None:
    """One bool per query, or None if verifyta could not check the model."""
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as f:
        f.write("\n".join(queries) + "\n")
        qfile = Path(f.name)
    try:
        raw = subprocess.run([VERIFYTA, "-q", str(xml), str(qfile)],
                             capture_output=True, text=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        return None
    finally:
        qfile.unlink()
    res = re.findall(r"Formula is (NOT )?satisfied", raw)
    return [r == "" for r in res] if len(res) == len(queries) else None


def original_queries(mission: str) -> list[str]:
    """The mission's original (repair-target) properties: its queries.q minus
    the goal-reachability queries added by make_goal_dataset.py."""
    d = GOALS / mission
    goals = {l.strip() for l in (d / "goal_queries.q").read_text().splitlines() if l.strip()}
    return [l.strip() for l in (d / "queries.q").read_text().splitlines()
            if l.strip() and l.strip() not in goals]


def evaluate_mission(run_dir: Path, model: str, mission: str, status: str) -> dict:
    orig  = original_queries(mission)
    goals = read_queries(GOALS / mission / "goal_queries.q")
    repaired = run_dir / model / mission / "repaired.xml"

    row = {"model": model, "mission": mission, "status": status,
           "n_orig": len(orig), "orig_passing": 0, "goal_preserved": True,
           "audited": True, "note": ""}

    if status in ("FIXED", "PARTIAL") and not repaired.exists():
        # The run reports a repair but did not save it: cannot be re-verified.
        row.update(audited=False, note="repaired.xml missing")
        return row
    final = repaired if repaired.exists() else GOALS / mission / "faulty.xml"
    fix = CORRECTIONS.get((run_dir.name.removeprefix("results_"), model, mission))
    if fix:
        # documented pipeline-defect correction: re-verify the candidate the
        # defect wrongly discarded (see corrections.json)
        final = run_dir / model / mission / fix["final_model"]
        row["note"] = "corrected: " + fix["reason"].split(":")[0]

    verdicts = verify(final, orig + goals)
    if verdicts is None:
        row.update(orig_passing=0, goal_preserved=False, joint_goal_kept=False,
                   note="verifyta could not check model")
        return row
    goal_ok = dict(zip(goals, verdicts[len(orig):]))
    row["orig_passing"]    = sum(verdicts[:len(orig)])
    row["goal_preserved"]  = all(ok for g, ok in goal_ok.items() if "&&" not in g)
    row["joint_goal_kept"] = all(ok for g, ok in goal_ok.items() if "&&" in g)
    row["broken_goals"]    = [g for g, ok in goal_ok.items() if not ok]
    return row


def summarize(rows: list[dict]) -> dict:
    aud = [r for r in rows if r["audited"]]
    n_q = sum(r["n_orig"] for r in aud)
    full = [r for r in aud if r["orig_passing"] == r["n_orig"]]
    pct = lambda a, b: round(100 * a / b, 2) if b else None
    return {
        "n_missions":        len(rows),
        "n_audited":         len(aud),
        "orig_passing":      sum(r["orig_passing"] for r in aud),
        "orig_total":        n_q,
        "orig_pass_rate":    pct(sum(r["orig_passing"] for r in aud), n_q),
        "goal_preserved":    sum(r["goal_preserved"] for r in aud),
        "joint_goal_kept":   sum(r.get("joint_goal_kept", True) for r in aud),
        "gated_pass_rate":   pct(sum(r["orig_passing"] for r in aud if r["goal_preserved"]), n_q),
        "genuine_fixes":     sum(r["goal_preserved"] for r in full),
        "genuine_fix_rate":  pct(sum(r["goal_preserved"] for r in full), len(aud)),
        "vacuous_fixes":     sum(not r["goal_preserved"] for r in full),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--include-original", action="store_true",
                    help="Also score the original runs (../results, ../results_no_trace); "
                         "output goes to evaluation/vs_original/ instead.")
    args = ap.parse_args()
    global OUT
    runs = own_runs()
    if args.include_original:
        runs = {**ORIGINAL_RUNS, **runs}
        OUT = OUT / "vs_original"
    if not runs:
        sys.exit("No goal_reachability/results_* folders to evaluate yet.")

    jobs = []
    for run, rdir in runs.items():
        if not rdir.exists():
            print(f"skip {run}: {rdir} not found")
            continue
        for f in sorted(rdir.glob("*_results.json")):
            model = f.name[: -len("_results.json")]
            if model in EXCLUDED:
                continue
            for r in json.loads(f.read_text()):
                jobs.append((run, rdir, model, r["mission"], r["status"]))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda j: {"run": j[0], **evaluate_mission(*j[1:])}, jobs))

    summary = {}
    for run in runs:
        for model in sorted({r["model"] for r in rows if r["run"] == run}):
            summary.setdefault(run, {})[model] = summarize(
                [r for r in rows if r["run"] == run and r["model"] == model])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_mission.json").write_text(json.dumps(rows, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["run", "model", "n_missions", "n_audited", "orig_passing", "orig_total",
              "orig_pass_rate", "goal_preserved",
              "joint_goal_kept", "gated_pass_rate", "genuine_fixes", "genuine_fix_rate", "vacuous_fixes"]
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for run, models in summary.items():
            for model, s in models.items():
                w.writerow({"run": run, "model": model, **s})

    # Headline metric: orig_pass_rate — original properties passing / all
    # original properties. Judging by it alone is valid only while every final
    # model keeps its goals; that is checked here on every run.
    for run, models in summary.items():
        print(f"\n== {run} ==")
        print(f"  {'model':<20} {'orig_pass_rate':>14} {'passing':>9} {'goal kept':>10}")
        for model, s in sorted(models.items(), key=lambda kv: -(kv[1]['orig_pass_rate'] or 0)):
            note = "" if s["n_audited"] == s["n_missions"] else \
                f"  ({s['n_missions'] - s['n_audited']} unaudited)"
            print(f"  {model:<20} {s['orig_pass_rate'] or 0:>13.2f}% "
                  f"{s['orig_passing']:>4}/{s['orig_total']:<4} "
                  f"{s['goal_preserved']:>5}/{s['n_audited']}{note}")
        broken = {m: s for m, s in models.items() if s["goal_preserved"] < s["n_audited"]}
        if broken:
            print(f"  WARNING: goals broken in {', '.join(broken)} — orig_pass_rate alone "
                  f"overstates these models; use gated_pass_rate in summary.json")
    print(f"\nWrote {OUT.relative_to(ROOT)}/summary.csv, summary.json, per_mission.json")


if __name__ == "__main__":
    main()
