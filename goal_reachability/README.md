# Goal reachability experiment

Everything for the goal-reachability experiment lives in this folder, in two
conditions — **no trace** (the LLM sees only the failing properties) and
**with trace** (it also sees verifyta's counterexample for each one). The
repair engine (`llm_fixer*.py`, `benchmark.py`, `benchmark_no_trace.py`) is
shared with the original experiment and stays in the repo root; nothing here
writes outside this folder.

The dataset is intentionally a top-level sibling to this experiment folder so the
paper can refer to the benchmark itself as a standalone artifact.

| Path | Contents |
|---|---|
| `../dataset/` | the 30 missions plus goal-reachability queries (see its README) |
| `make_goal_dataset.py` | documents how the goal queries were generated from the original benchmark (not included) |
| `check_api_keys.py` | pre-flight check: keys set, keys accepted, model ids available |
| `run_no_trace.sh` | no-trace benchmark → analysis → figures → fair evaluation |
| `run_trace.sh` | with-trace benchmark → analysis → figures → fair evaluation |
| `evaluate_goal_preserving.py` | fair scoring of every run (see below) |
| `make_comparison_figures.py` | trace vs no-trace figures + `comparison_table.csv` → `figures_comparison/` |
| `results_no_trace/`, `results_trace/` | created by the runs (per-model results + `analytics.json`) |
| `figures_no_trace/`, `figures_trace/` | created by the runs |
| `archive/` | earlier runs superseded by a dataset change (not evaluated) |
| `evaluation/` | evaluator output for this experiment (`vs_original/` only with `--include-original`) |

## Running (from the repo root)

```bash
source ~/.llm_api_keys                                # the export lines for your keys

python goal_reachability/check_api_keys.py            # add --ping to test one real call per model
bash   goal_reachability/run_no_trace.sh --models a,b,c   # comma-separated, no spaces
bash   goal_reachability/run_trace.sh    --models a,b,c   # same, with counterexample traces
bash   goal_reachability/run_no_trace.sh --models a --resume   # (works for run_trace.sh too)
```

Both scripts use the same dataset, repair loop, retry policy and regression
check — only the evidence in the prompt differs — so the two conditions are
directly comparable. The evaluator prints one section per condition
(`goal_reachability_no_trace`, `goal_reachability_trace`).

`--resume` keeps finished missions and re-runs the ones lost to the provider
(no credit, quota, rate limit, outage). A run stops at the first key / credit /
quota error and prints which mission to continue from.

## Metric used in this experiment: `orig_pass_rate`

Every model in this experiment is judged by **`orig_pass_rate`** = original
properties passing ÷ all original properties (60 = 30 missions × 2, all
failing before repair). It is in `evaluation/summary.json` (with the raw count
in `orig_passing` / `orig_total`) and is what the comparison figures plot.

This is valid because the goal queries are always preserved here: the harness
rejects any attempt that breaks a goal, so every final model keeps them
(`goal kept` = 30/30 for every model). The evaluator re-checks this on every
run and prints a WARNING if a final model ever breaks a goal — in that case
use `gated_pass_rate` instead.

Do not use `query_pass_rate` from `analytics.json`: it also counts the 2 goal
queries per mission, which always pass, so it equals 50% + orig_pass_rate / 2.

## Fair evaluation

The goal queries pass on the faulty model before any repair, so they must never
earn credit. `evaluate_goal_preserving.py` re-verifies the final model of every
model × mission in every `goal_reachability/results_*` run and scores them the
same way (output: `evaluation/`). The original runs in the repo root are only
scored with `--include-original`, into a separate `evaluation/vs_original/`:

- **credit** comes only from the 2 original properties per mission
- **gate**: that credit counts only if every per-robot goal (`E<> robot.Goal`)
  is still reachable in the final model

The **final model** is the run's `repaired.xml`, or `faulty.xml` when the run
saved no repair (a FAILED mission — it scores 0, since all original properties
fail there). Every final model is re-checked with verifyta; the run's own JSON
is not trusted.

```bash
python goal_reachability/evaluate_goal_preserving.py                     # this experiment only
python goal_reachability/evaluate_goal_preserving.py --include-original  # + original runs, for comparison
```

### Reading the output

```
  model                orig_pass_rate   passing  goal kept
  gpt-5.1                      38.33%   23/60      30/30
```

The printout shows the headline metric and the goal check; every field below
is in `summary.json` / `summary.csv`.

Per model, over its 30 missions × 2 original properties = 60 properties:

| Column | Field in `summary.json` | Calculated as | Meaning |
|---|---|---|---|
| **orig pass** | `orig_pass_rate` | original properties passing in the final model ÷ 60 | Ungated score — what the original benchmark measured. Inflated when repairs break the mission. |
| **gated pass** | `gated_pass_rate` | original properties passing, **counted only in missions where goal kept**, ÷ 60 | **Headline metric (property level).** A mission that passes its properties by breaking the goal contributes 0. |
| **genuine fix** | `genuine_fixes` / `genuine_fix_rate` | missions where **both** original properties pass **and** the goal is kept, out of the audited missions | **Headline metric (mission level).** Fully and correctly repaired missions. |
| **vacuous** | `vacuous_fixes` | missions where both original properties pass **but** a goal is broken | "Fixes" that only work by making the mission unable to finish, e.g. a time-lock. The original benchmark counted these as FIXED. |
| **goal kept** | `goal_preserved` | missions whose final model still satisfies every `E<> robot.Goal` | How often the model left the mission able to complete. Includes FAILED missions, where the unchanged faulty model trivially keeps the goal. |

Other fields in `summary.json`:

- `n_missions` — missions in the model's `<model>_results.json`
- `n_audited` — missions whose final model could be re-verified. A mission
  reported FIXED/PARTIAL without a saved `repaired.xml` cannot be checked and
  is left out of every rate (the table then shows "(N unaudited)")
- `joint_goal_kept` — only meaningful for datasets built with
  `make_goal_dataset.py --joint`; with the default dataset it always equals
  `n_audited`

Relations that always hold: `gated pass ≤ orig pass`, and
`genuine fix + vacuous` = missions with both original properties passing.

**In this experiment** the goal queries are part of the model, so the harness
rejects any attempt that breaks a goal as a regression. Final models therefore
always keep the goal: `goal kept` = 30/30, `vacuous` = 0 and
`gated pass` = `orig pass`. The columns only diverge when scoring the original
runs (`--include-original`) — e.g. gpt-5.1's original no-trace run: 83.3%
orig pass but 21.7% gated, with 18 vacuous fixes.

## Comparing with the original no-trace run

Use the evaluator's numbers, not `analytics.json`.

`query_pass_rate` in `analytics.json` is over 4 queries per mission (2 original
+ 2 goal) instead of 2, and the goal queries almost always pass, so it is
inflated and not comparable to `../results_no_trace/`. `fix_rate`
is comparable, and now means "all original properties pass and the mission
goal is still reachable". The LLM also sees the goal queries in the model and
vacuous attempts are rejected as regressions, so treat this as a separate
experimental condition.

## Trace vs no-trace figures

`make_comparison_figures.py` (also run at the end of both run scripts) counts,
for every model with results in both conditions, how many of the **60
repair-target queries** (30 missions × 2 original properties, all failing
before repair) pass after repair, with and without the trace. The 60 goal
queries are not counted: they pass in every run, so they would add the same 60
to every bar.

| Figure | Shows |
|---|---|
| `fig_cmp1_queries_passed` | queries repaired per model (x/60), with vs without trace |
| `fig_cmp2_queries_delta` | per model, queries gained and lost with the trace, and the net change |
| `fig_cmp3_by_fault_category` | queries repaired per fault category, all models pooled, with vs without trace |
| `fig_cmp4_category_heatmap` | model × fault category: queries gained (+) / lost (−) with the trace |
| `queries_passed.csv` | every count above |
