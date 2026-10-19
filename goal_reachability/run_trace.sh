#!/usr/bin/env bash
# With-trace benchmark on the goal-reachability dataset (the LLM also sees the
# verifyta counterexample for each failing property). Everything this run
# produces stays inside goal_reachability/.
#
#   bash goal_reachability/run_trace.sh --models gpt-4o,gpt-5-mini   # comma-separated, no spaces
#   bash goal_reachability/run_trace.sh --models gpt-4o --resume      # continue an interrupted run
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f venv/bin/activate ] && source venv/bin/activate   # optional local venv

HERE=goal_reachability
DATASET=dataset
python benchmark.py \
  --missions-dir "$DATASET" \
  --results-dir  "$HERE/results_trace" \
  "$@"

python analyze_results.py --dir "$HERE/results_trace"
python make_figures.py --dir "$HERE/results_trace" --out "$HERE/figures_trace"

# Fair scoring across all goal_reachability runs (goal queries gate, never earn credit)
python "$HERE/evaluate_goal_preserving.py"

# Trace vs no-trace comparison figures (skipped until both conditions have results)
python "$HERE/make_comparison_figures.py" || true
