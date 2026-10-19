#!/usr/bin/env bash
# No-trace benchmark on the goal-reachability dataset. Everything this run
# produces stays inside goal_reachability/.
#
#   bash goal_reachability/run_no_trace.sh                      # all models
#   bash goal_reachability/run_no_trace.sh --models gpt-4o      # extra args go to the benchmark
#   bash goal_reachability/run_no_trace.sh --resume             # continue an interrupted run
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate   # optional local venv

HERE=goal_reachability
DATASET=dataset
python benchmark_no_trace.py \
  --missions-dir "$DATASET" \
  --results-dir  "$HERE/results_no_trace" \
  "$@"

python analyze_results.py --dir "$HERE/results_no_trace"
python make_figures.py --dir "$HERE/results_no_trace" --out "$HERE/figures_no_trace"

# Fair scoring across all runs (goal queries gate, never earn credit)
python "$HERE/evaluate_goal_preserving.py"

# Trace vs no-trace comparison figures (skipped until both conditions have results)
python "$HERE/make_comparison_figures.py" || true
