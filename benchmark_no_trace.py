"""
benchmark_no_trace.py

Ablation counterpart to benchmark.py: runs the exact same missions/ dataset
and the exact same repair loop (verify → prompt → LLM → validate → re-verify
→ regression check → retry), but via llm_fixer_no_trace.build_prompt_no_trace,
which never shows the LLM the counterexample trace verifyta produced — only
the failing formula and its description. Results are written to a separate
results_no_trace/ folder so they never mix with the trace-based run and can
be diffed head-to-head per mission/model.

Usage:
    # Required API keys (set whichever models you run):
    export OPENAI_API_KEY='sk-...'
    export ANTHROPIC_API_KEY='sk-ant-...'
    export XAI_API_KEY='xai-...'
    export DEEPSEEK_API_KEY='sk-...'
    export GOOGLE_API_KEY='AIza...'

    python benchmark_no_trace.py                          # all models, all missions
    python benchmark_no_trace.py --models gpt-4o,gpt-5.1  # specific models
    python benchmark_no_trace.py --mission 01_warehouse_inspection
    python benchmark_no_trace.py --dry-run                # verify dataset, no LLM calls

    Available model names:
      gpt-5.1  gpt-5-mini  gpt-5-nano  gpt-4o
      claude-sonnet-4-5  claude-haiku-4-5
      grok-code-fast  grok-3-mini
      deepseek-reasoner  deepseek-r1:32b  deepseek-coder:33b  qwen3.5:9b
      gemini-2.0-flash  gemini-2.5-pro
"""

import argparse
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pyuppaal

import llm_fixer
from llm_fixer import (
    VERIFYTA, LLM_MODEL, MAX_REPAIR_ATTEMPTS, MODELS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY,
    DEEPSEEK_API_KEY, GOOGLE_API_KEY,
    verify_model, call_llm, extract_xml, check_regressions,
    C, ok, err, warn, info, dim, header,
)
from llm_fixer_no_trace import build_prompt_no_trace
from pyuppaal import UModel

ALL_MODELS = list(MODELS.keys())

_PROVIDER_KEY_MAP = {
    "openai":    lambda: OPENAI_API_KEY,
    "anthropic": lambda: ANTHROPIC_API_KEY,
    "xai":       lambda: XAI_API_KEY,
    "deepseek":  lambda: DEEPSEEK_API_KEY,
    "google":    lambda: GOOGLE_API_KEY,
    "ollama":    lambda: "local",
}

MISSIONS_DIR  = Path(__file__).parent / "dataset"
RESULTS_BASE  = Path(__file__).parent / "results_no_trace"
RESULTS_FILE  = Path(__file__).parent / "benchmark_no_trace_results.json"


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class AttemptRecord:
    attempt:     int
    n_failing:   int
    n_passing:   int
    regressions: list[int]
    llm_valid:   bool        # did LLM return parseable XML?
    # served_model + token counts reported by the provider (llm_fixer.LAST_CALL);
    # empty when the call failed or for runs made before this was recorded
    usage:       dict = field(default_factory=dict)

@dataclass
class MissionResult:
    mission:          str
    n_queries:        int
    initial_failing:  list[str]   # query strings
    initial_passing:  list[int]
    status:           str         # FIXED | PARTIAL | FAILED | ERROR | SKIP
    attempts_used:    int
    final_failing:    list[str]
    final_passing:    list[str]
    has_ground_truth: bool
    attempts:         list[AttemptRecord] = field(default_factory=list)
    error_msg:        str = ""
    elapsed_s:        float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def status_colored(status: str) -> str:
    colors = {
        "FIXED":   C.GREEN,
        "PARTIAL": C.YELLOW,
        "FAILED":  C.RED,
        "ERROR":   C.MAGENTA,
        "SKIP":    C.DIM,
    }
    c = colors.get(status, "")
    return f"{C.BOLD}{c}{status:<7}{C.RESET}"


def print_summary(results: list[MissionResult]) -> None:
    total   = len(results)
    fixed   = sum(1 for r in results if r.status == "FIXED")
    partial = sum(1 for r in results if r.status == "PARTIAL")
    failed  = sum(1 for r in results if r.status == "FAILED")
    skipped = sum(1 for r in results if r.status == "SKIP")
    errors  = sum(1 for r in results if r.status == "ERROR")
    avg_att = (sum(r.attempts_used for r in results if r.status not in ("SKIP", "ERROR")) /
               max(1, fixed + partial + failed))

    header("BENCHMARK SUMMARY (no-trace)")
    col_w = [28, 9, 8, 10, 8]
    hdr = (f"{'Mission':<{col_w[0]}}  {'Status':<{col_w[1]}}"
           f"  {'Queries':<{col_w[2]}}  {'Attempts':<{col_w[3]}}"
           f"  {'GT?':<{col_w[4]}}")
    print(f"  {C.BOLD}{hdr}{C.RESET}")
    print(f"  {'─'*sum(col_w)}")
    for r in results:
        gt = f"{C.GREEN}yes{C.RESET}" if r.has_ground_truth else f"{C.DIM}no{C.RESET}"
        q_info = f"{len(r.final_passing)}/{r.n_queries} pass"
        print(
            f"  {r.mission:<{col_w[0]}}  {status_colored(r.status)}"
            f"  {q_info:<{col_w[2]+2}}  {r.attempts_used:<{col_w[3]}}  {gt}"
        )
    print(f"\n  {C.BOLD}Total:{C.RESET} {total}  "
          f"{C.GREEN}Fixed: {fixed}{C.RESET}  "
          f"{C.YELLOW}Partial: {partial}{C.RESET}  "
          f"{C.RED}Failed: {failed}{C.RESET}  "
          f"{C.MAGENTA}Errors: {errors}{C.RESET}  "
          f"{C.DIM}Skipped: {skipped}  Avg attempts: {avg_att:.1f}{C.RESET}")


# ── Per-mission runner ────────────────────────────────────────────────────────

# Provider errors that retrying cannot fix (bad key, no credit, no quota):
# the model's run stops at the first one instead of recording a fake FAILED
# for every remaining mission.
_FATAL_KEYS = ("credential", "api_key", "authenticate", "unauthorized", "quota",
               "permission", "credit balance", "billing", "insufficient balance",
               "payment")
# Signs that an attempt was lost to the provider rather than to the model's
# own output (verifyta errors start with "Command:" and are not included).
_API_ERROR_KEYS = _FATAL_KEYS + ("error code:", "rate limit", "overloaded",
                                 "unavailable", "timed out", "timeout",
                                 "connection")


def is_fatal_api_error(msg: str) -> bool:
    return any(k in msg.lower() for k in _FATAL_KEYS)


def is_api_error(msg: str) -> bool:
    return bool(msg) and any(k in msg.lower() for k in _API_ERROR_KEYS)


def run_mission(mission_dir: Path, dry_run: bool,
                results_dir: Path | None = None,
                model: str | None = None) -> MissionResult:
    name     = mission_dir.name
    xml_path = mission_dir / "faulty.xml"
    has_gt   = (mission_dir / "repaired.xml").exists()

    if not xml_path.exists():
        return MissionResult(
            mission=name, n_queries=0, initial_failing=[], initial_passing=[],
            status="SKIP", attempts_used=0, final_failing=[], final_passing=[],
            has_ground_truth=has_gt, error_msg="faulty.xml not found",
        )

    t0 = time.time()
    try:
        failures, passing, _ = verify_model(str(xml_path))
    except Exception as e:
        return MissionResult(
            mission=name, n_queries=0, initial_failing=[], initial_passing=[],
            status="ERROR", attempts_used=0, final_failing=[], final_passing=[],
            has_ground_truth=has_gt, error_msg=str(e), elapsed_s=time.time()-t0,
        )

    n_queries = len(failures) + len(passing)
    all_queries = UModel(str(xml_path)).queries
    result = MissionResult(
        mission=name, n_queries=n_queries,
        initial_failing=[f["query"] for f in failures],
        initial_passing=passing,
        status="SKIP", attempts_used=0,
        final_failing=[f["query"] for f in failures],
        final_passing=[all_queries[i-1] for i in passing],
        has_ground_truth=has_gt,
        elapsed_s=0.0,
    )

    if not failures:
        result.status        = "FIXED"   # already passing — nothing to fix
        result.final_passing = list(all_queries)
        result.elapsed_s     = time.time() - t0
        return result

    if dry_run:
        result.status = "SKIP"
        result.error_msg = "dry-run: LLM skipped"
        result.elapsed_s = time.time() - t0
        return result

    # ── LLM repair loop (no trace) ───────────────────────────────────────────
    src_path      = str(xml_path)
    cur_failures  = failures
    cur_passing   = passing
    workdir       = results_dir if results_dir else mission_dir / "_repair_no_trace"
    workdir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        xml_source = Path(src_path).read_text()
        prompt     = build_prompt_no_trace(xml_source, cur_failures)

        info(f"  Attempt {attempt}/{MAX_REPAIR_ATTEMPTS}: waiting for LLM ({model})…")
        llm_t0 = time.time()
        try:
            raw_reply = call_llm(prompt, model)
            usage     = dict(llm_fixer.LAST_CALL)
            (workdir / f"attempt{attempt}_raw.txt").write_text(raw_reply)
            fixed_xml = extract_xml(raw_reply)
        except Exception as e:
            dim(f"    LLM call failed after {time.time()-llm_t0:.1f}s")
            result.attempts.append(AttemptRecord(
                attempt=attempt, n_failing=len(cur_failures),
                n_passing=len(cur_passing), regressions=[], llm_valid=False,
            ))
            result.error_msg = str(e)
            # Auth / quota errors won't be fixed by retrying — stop immediately
            e_str = str(e).lower()
            if is_fatal_api_error(e_str):
                result.status = "ERROR"
                result.elapsed_s = time.time() - t0
                return result
            continue  # transient/unknown error — still burn an attempt and retry

        dim(f"    LLM responded in {time.time()-llm_t0:.1f}s ({len(raw_reply)} chars)")

        if not fixed_xml.startswith("<?xml"):
            warn(f"    LLM output did not start with <?xml — invalid, discarding attempt "
                 f"(raw saved to attempt{attempt}_raw.txt)")
            result.attempts.append(AttemptRecord(
                attempt=attempt, n_failing=len(cur_failures),
                n_passing=len(cur_passing), regressions=[], llm_valid=False,
                usage=usage,
            ))
            continue

        try:
            ET.fromstring(fixed_xml)
        except ET.ParseError:
            warn(f"    LLM output is not well-formed XML — invalid, discarding attempt "
                 f"(raw saved to attempt{attempt}_raw.txt)")
            result.attempts.append(AttemptRecord(
                attempt=attempt, n_failing=len(cur_failures),
                n_passing=len(cur_passing), regressions=[], llm_valid=False,
                usage=usage,
            ))
            continue

        fixed_path = workdir / f"attempt{attempt}.xml"
        fixed_path.write_text(fixed_xml)

        info(f"  Attempt {attempt}/{MAX_REPAIR_ATTEMPTS}: verifying with verifyta…")
        verify_t0 = time.time()
        try:
            new_failures, new_passing, _ = verify_model(str(fixed_path))
        except Exception as e:
            dim(f"    Verification failed after {time.time()-verify_t0:.1f}s")
            result.attempts.append(AttemptRecord(
                attempt=attempt, n_failing=len(cur_failures),
                n_passing=len(cur_passing), regressions=[], llm_valid=True,
                usage=usage,
            ))
            result.error_msg = str(e)
            continue
        dim(f"    Verified in {time.time()-verify_t0:.1f}s — "
            f"{len(new_failures)} failing / {len(new_passing)} passing")

        regressions = check_regressions(cur_passing, new_failures)
        result.attempts.append(AttemptRecord(
            attempt=attempt, n_failing=len(new_failures),
            n_passing=len(new_passing), regressions=regressions, llm_valid=True,
            usage=usage,
        ))

        if regressions:
            # Reject this attempt — don't advance src_path
            continue

        result.attempts_used = attempt

        if not new_failures:
            shutil.copy(str(fixed_path), str(workdir / "repaired.xml"))
            result.status        = "FIXED"
            result.final_failing = []
            result.final_passing = [UModel(str(fixed_path)).queries[i-1] for i in new_passing]
            result.elapsed_s     = time.time() - t0
            return result

        # Partial improvement — save best so far and advance
        shutil.copy(str(fixed_path), str(workdir / "repaired.xml"))
        src_path     = str(fixed_path)
        cur_failures = new_failures
        cur_passing  = new_passing

    # Exhausted attempts (or stopped early on a hard failure) — report the
    # real number of attempts actually made, not the configured maximum.
    result.attempts_used = attempt
    result.final_failing = [f["query"] for f in cur_failures]
    result.final_passing = [UModel(src_path).queries[i-1] for i in cur_passing]
    result.status        = "PARTIAL" if len(cur_failures) < len(failures) else "FAILED"
    result.elapsed_s     = time.time() - t0
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_keys(models: list[str]) -> None:
    from llm_fixer import MODELS as MODEL_REG
    missing = set()
    for m in models:
        provider = MODEL_REG.get(m, ("openai", m))[0]
        key = _PROVIDER_KEY_MAP.get(provider, lambda: "")()
        if not key:
            missing.add(f"{provider.upper()}_API_KEY  (needed for {m})")
    for msg in sorted(missing):
        warn(f"Missing env var: {msg}")


def run_one_model(
    model: str,
    dirs: list[Path],
    dry_run: bool,
    resume: bool = False,
) -> tuple[list[MissionResult], Path | None]:
    model_label     = model.replace("/", "-").replace(":", "-")
    llm_results_dir = (RESULTS_BASE / model_label) if not dry_run else None
    if llm_results_dir:
        llm_results_dir.mkdir(parents=True, exist_ok=True)

    results: list[MissionResult] = []
    for i, mission_dir in enumerate(dirs, 1):
        mission_results_dir = (llm_results_dir / mission_dir.name) if llm_results_dir else None

        # ── resume: reload saved result and skip re-running ─────────────────
        if resume and mission_results_dir:
            saved = mission_results_dir / "result.json"
            # also treat old runs (no result.json) as done if attempt files exist
            already_ran = saved.exists() or (
                mission_results_dir.exists() and
                any(mission_results_dir.glob("attempt*.xml"))
            )
            if already_ran and saved.exists():
                prev = json.loads(saved.read_text())
                if prev["status"] == "ERROR" or is_api_error(prev.get("error_msg", "")):
                    # Lost to the provider (no credit, outage, rate limit...),
                    # not a real result: discard it and run the mission again.
                    warn(f"[{i}/{len(dirs)}] {mission_dir.name}: previous run hit an API "
                         f"error — re-running")
                    shutil.rmtree(mission_results_dir)
                    already_ran = False
            if already_ran:
                if saved.exists():
                    r = MissionResult(**json.loads(saved.read_text()))
                    r.attempts = [AttemptRecord(**a) if isinstance(a, dict) else a
                                  for a in r.attempts]
                    status_str = status_colored(r.status)
                else:
                    # old run: no result.json — show as DONE without details
                    status_str = f"{C.DIM}DONE   {C.RESET}"
                    r = None
                print(f"\n{C.BOLD}{C.BLUE}[{i}/{len(dirs)}]{C.RESET} "
                      f"{C.BOLD}{mission_dir.name}{C.RESET}  "
                      f"{C.DIM}(resumed — {status_str}{C.DIM}){C.RESET}")
                if r:
                    results.append(r)
                continue

        print(f"\n{C.BOLD}{C.BLUE}[{i}/{len(dirs)}]{C.RESET} {C.BOLD}{mission_dir.name}{C.RESET}")
        r = run_mission(mission_dir, dry_run=dry_run,
                        results_dir=mission_results_dir, model=model)
        results.append(r)

        print(f"  Status   : {status_colored(r.status)}  ({r.elapsed_s:.1f}s)")
        print(f"  Attempts : {r.attempts_used}/{MAX_REPAIR_ATTEMPTS}")
        if r.final_failing:
            print(f"  {C.RED}Failing  :{C.RESET}")
            for q in r.final_failing:
                print(f"    {C.RED}✗{C.RESET} {q}")
        else:
            print(f"  {C.GREEN}Failing  : none{C.RESET}")
        if r.final_passing:
            print(f"  {C.GREEN}Passing  :{C.RESET}")
            for q in r.final_passing:
                print(f"    {C.GREEN}✓{C.RESET} {q}")
        if r.attempts:
            parts = [f"#{a.attempt}:{a.n_failing}fail/{a.n_passing}pass"
                     + (" REG" if a.regressions else "")
                     for a in r.attempts]
            print(f"  {C.DIM}Progress :   " + "  →  ".join(parts) + C.RESET)
        if r.error_msg and r.status not in ("SKIP",):
            warn(f"  {r.error_msg}")
        if mission_results_dir and r.status in ("FIXED", "PARTIAL"):
            dim(f"  Saved → {RESULTS_BASE.name}/{model_label}/{mission_dir.name}/")

        # persist per-mission result for future --resume
        if mission_results_dir and not dry_run:
            mission_results_dir.mkdir(parents=True, exist_ok=True)
            (mission_results_dir / "result.json").write_text(
                json.dumps(asdict(r), indent=2)
            )

        if r.status == "ERROR" and is_fatal_api_error(r.error_msg):
            results.pop()   # not a real result; --resume will re-run it
            err(f"  Stopping {model}: the provider rejected the request "
                f"(key / credit / quota). Fix it, then re-run with --resume "
                f"to continue from {mission_dir.name}.")
            break

    return results, llm_results_dir


def print_cross_model_summary(
    all_results: dict[str, list[MissionResult]],
    dirs: list[Path],
) -> None:
    models = list(all_results.keys())
    col_w  = max(len(m) for m in models) + 2
    row_w  = max(len(d.name) for d in dirs) + 2

    header("CROSS-MODEL SUMMARY (no-trace)")
    print(f"  {'':<{row_w}}", end="")
    for m in models:
        print(f"  {m:<{col_w}}", end="")
    print()
    print(f"  {'─'*(row_w + (col_w+2)*len(models))}")

    sym = {"FIXED":   f"{C.GREEN}FIXED  {C.RESET}",
           "PARTIAL": f"{C.YELLOW}PARTIAL{C.RESET}",
           "FAILED":  f"{C.RED}FAILED {C.RESET}",
           "ERROR":   f"{C.MAGENTA}ERROR  {C.RESET}",
           "SKIP":    f"{C.DIM}SKIP   {C.RESET}"}

    for d in dirs:
        print(f"  {d.name:<{row_w}}", end="")
        for m in models:
            r = next((r for r in all_results[m] if r.mission == d.name), None)
            print(f"  {sym.get(r.status if r else 'SKIP', '?'):<{col_w}}", end="")
        print()

    print(f"\n  {'FIXED / total':<{row_w}}", end="")
    for m in models:
        fixed = sum(1 for r in all_results[m] if r.status == "FIXED")
        total = len(dirs)
        print(f"  {C.BOLD}{fixed}/{total}{C.RESET}{' '*(col_w-len(str(fixed))-len(str(total))-1)}", end="")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions-dir", default=str(MISSIONS_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mission", default=None,
                        help="Single mission folder name.")
    parser.add_argument("--models", default="all",
                        help=f"Comma-separated model names or 'all'. "
                             f"Available: {', '.join(ALL_MODELS)}")
    parser.add_argument("--resume", action="store_true",
                        help="Skip missions that already have a saved result.json.")
    parser.add_argument("--results-dir", default=None,
                        help="Output folder (default: results_no_trace/). When set, "
                             "the combined JSON is also written inside it (all_models_combined.json), so the "
                             "run stays fully separate from earlier ones.")
    args = parser.parse_args()

    global RESULTS_BASE, RESULTS_FILE
    if args.results_dir:
        RESULTS_BASE = Path(args.results_dir)
        # Not "*_results.json": analyze_results.py / make_figures.py treat
        # every file matching that pattern as one model's results.
        RESULTS_FILE = RESULTS_BASE / "all_models_combined.json"
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    pyuppaal.set_verifyta_path(VERIFYTA)

    # resolve model list
    if args.models.strip().lower() == "all":
        selected_models = ALL_MODELS
    else:
        selected_models = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in selected_models if m not in MODELS]
        if unknown:
            err(f"Unknown model(s): {unknown}")
            info(f"Available: {ALL_MODELS}")
            sys.exit(1)

    if not args.dry_run:
        _check_keys(selected_models)

    missions_dir = Path(args.missions_dir)
    if not missions_dir.exists():
        err(f"Missions directory not found: {missions_dir}")
        sys.exit(1)

    dirs = ([missions_dir / args.mission] if args.mission
            else sorted(p for p in missions_dir.iterdir()
                        if p.is_dir() and (p / "faulty.xml").exists()))
    if not dirs:
        err("No mission directories with faulty.xml found.")
        sys.exit(1)

    info(f"Models   : {C.BOLD}{', '.join(selected_models)}{C.RESET}")
    info(f"Missions : {len(dirs)}   Attempts/mission: {MAX_REPAIR_ATTEMPTS}")
    info(f"Mode     : {C.BOLD}no-trace{C.RESET} (counterexample traces withheld from the LLM)")
    info(f"Missions dir: {C.BOLD}{missions_dir}{C.RESET}   Output: {C.BOLD}{RESULTS_BASE}{C.RESET}")
    if args.resume:
        info(f"Resume   : {C.BOLD}on{C.RESET} (skipping already-completed missions)")

    # ── run each model ───────────────────────────────────────────────────────
    all_results: dict[str, list[MissionResult]] = {}
    combined = []

    for model in selected_models:
        mode = "DRY-RUN" if args.dry_run else f"LLM={model}"
        header(f"BENCHMARK (no-trace)  [{mode}]  {len(dirs)} missions")

        results, _ = run_one_model(model, dirs, args.dry_run, resume=args.resume)
        all_results[model] = results
        print_summary(results)
        for r in results:
            combined.append({"model": model, **asdict(r)})

        # per-model JSON
        if not args.dry_run:
            model_label = model.replace("/", "-").replace(":", "-")
            out = RESULTS_BASE / f"{model_label}_results.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps([asdict(r) for r in results], indent=2))
            dim(f"  Saved → {out.name}")

    if len(selected_models) > 1:
        print_cross_model_summary(all_results, dirs)

    RESULTS_FILE.write_text(json.dumps(combined, indent=2))
    info(f"Combined results → {C.BOLD}{RESULTS_FILE.name}{C.RESET}")


if __name__ == "__main__":
    main()
