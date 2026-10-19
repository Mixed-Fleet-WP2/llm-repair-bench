# LLM-based repair of UPPAAL mission models

This repository contains the benchmark dataset and the scripts used to compare
repair quality with and without the counterexample trace from UPPAAL.

## Layout

```text
dataset/
  benchmark models, properties, and goal-reachability checks

goal_reachability/
  run_no_trace.sh, run_trace.sh
  evaluate_goal_preserving.py
  make_comparison_figures.py
  results_no_trace/, results_trace/
  evaluation/
  figures_comparison/
  token_probe/
  check_api_keys.py

benchmark.py
benchmark_no_trace.py
llm_fixer.py
llm_fixer_no_trace.py
analyze_results.py
make_figures.py
```

The dataset is stored at the repo root so it is treated as a standalone artifact,
while `goal_reachability/` contains the experiment logic and outputs.

## Requirements

- Python 3.12
- `pip install -r requirements.txt`
- UPPAAL 5.x with `verifyta` installed separately
- API keys for the LLMs you want to run
- Optional: Ollama for Qwen3.5-9B

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `VERIFYTA` to the installed binary:

```bash
export VERIFYTA=/path/to/uppaal-5.x/bin/verifyta
```

## API keys

The simplest method is to store them in a local shell file and source it:

```bash
cat > ~/.llm_api_keys <<'EOF'
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
EOF

source ~/.llm_api_keys
```

You only need the keys for the providers/models you will actually run.

## Quick start

From the repo root:

```bash
export VERIFYTA=/path/to/uppaal-5.x/bin/verifyta
source ~/.llm_api_keys

python goal_reachability/check_api_keys.py
python goal_reachability/check_api_keys.py --models gpt-5-mini,claude-haiku-4-5 --ping
```

The final check performs a tiny real API call to confirm the model is reachable and
that the provider is serving the expected model.

## Reproduce evaluation and figures

This uses the included raw results and does not call external LLM APIs:

```bash
export VERIFYTA=/path/to/uppaal-5.x/bin/verifyta
python goal_reachability/evaluate_goal_preserving.py
python goal_reachability/make_comparison_figures.py
```

Outputs go to:

- `goal_reachability/evaluation/summary.csv`
- `goal_reachability/evaluation/summary.json`
- `goal_reachability/evaluation/per_mission.json`
- `goal_reachability/figures_comparison/`

## Re-run the benchmark

This actually queries the LLMs and may incur API costs:

```bash
source ~/.llm_api_keys
export VERIFYTA=/path/to/uppaal-5.x/bin/verifyta

bash goal_reachability/run_no_trace.sh --models gpt-5-mini,claude-haiku-4-5
bash goal_reachability/run_trace.sh    --models gpt-5-mini,claude-haiku-4-5
```

Useful options:

```bash
bash goal_reachability/run_no_trace.sh --mission 08_construction_lift
bash goal_reachability/run_trace.sh    --mission 08_construction_lift
bash goal_reachability/run_no_trace.sh --resume
bash goal_reachability/run_trace.sh    --resume
```

`--resume` continues an interrupted run without redoing finished missions.

The benchmark rejects repairs that satisfy the violated properties only by
breaking goal reachability, so the goal queries act as a validity check rather
than as credit.

