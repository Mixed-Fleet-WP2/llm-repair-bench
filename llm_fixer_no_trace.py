"""
llm_fixer_no_trace.py

Ablation variant of llm_fixer.py: verifies a UPPAAL model, collects all
failing queries, and asks the LLM to fix them WITHOUT ever showing it the
counterexample trace produced by verifyta — only the failing formula and its
description are given, plus a note that no trace is available. Everything
else (provider routing, XML validation, regression checking, retry loop) is
reused unchanged from llm_fixer.py so the two pipelines only differ in the
one thing being ablated: trace evidence.

Used to compare "LLM repairs with a concrete counterexample" against
"LLM repairs blind, from the property text alone" — see benchmark_no_trace.py.
"""

import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

from llm_fixer import (
    VERIFYTA, MAX_REPAIR_ATTEMPTS,
    call_llm, extract_xml, check_regressions, verify_model, log_diff, setup,
    C, ok, err, warn, info, dim, header,
)

MODEL_FILE = "faulty_warehouse.xml"
FIXED_FILE = "fixed_warehouse_no_trace.xml"

# ── Prompt construction (no-trace variant) ───────────────────────────────────
# Identical to llm_fixer.UPPAAL_PRIMER except rule 4, reworded because it's
# the only rule that assumes an execution trace is available to walk through.

UPPAAL_PRIMER_NO_TRACE = textwrap.dedent("""
    ═══ UPPAAL SEMANTICS YOU MUST RESPECT ═══

    1. GUARD SCOPE: A guard, invariant, or assignment can only reference
       variables, clocks, and constants visible to that specific template
       (global declarations + that template's own local declarations). A guard
       CANNOT reference another process's current location using
       `ProcessInstance.LocationName` syntax (e.g. `!UGV.Inspecting` is ILLEGAL
       inside a <label kind="guard">). That syntax only works inside <query>
       formulas. If a fix needs one process to react to another process's
       state, the correct pattern is: the OTHER process sets/clears a shared
       global boolean on its own transitions, and you guard on that boolean.

    2. CHANNEL TYPES — get this wrong and the fix will silently do nothing:
       - `chan c` (binary rendezvous): sender and receiver only fire together
         if BOTH have an enabled edge (guard true) at the same instant.
       - `urgent chan c`: same as above, but the model may not let time pass
         while the edge is enabled — forces immediate synchronization.
       - `broadcast chan c`: the sender fires the instant its OWN guard is
         true, REGARDLESS of whether any receiver is ready. A receiver whose
         guard is false at that exact instant simply misses the broadcast and
         does not transition — and if that broadcast is only ever sent once,
         the receiver can be permanently stuck. When editing a guard on a
         broadcast receiver, always confirm the guard is actually true at the
         moment the corresponding sender fires.

    3. COMMITTED locations force ONE outgoing edge to fire with ZERO elapsed
       time before anything else in the system can happen. Two committed
       locations chained together will always fire back-to-back instantly —
       this is a very common root cause of "two actions look simultaneous"
       bugs. The fix is a real guard on the second edge that depends on a
       completion flag set by the first action, not just reordering states.

    4. GUARD-DIRECTION BUGS ARE THE #1 CAUSE OF FAILURES YOU WILL SEE. No
       execution trace is available for this task, so reason about it
       structurally instead: for each failing property, enumerate the
       transitions in the model that plausibly interact with the locations,
       clocks, and variables named in the query, and mentally simulate the
       small number of orderings between processes that could make the
       property false. Watch specifically for a guard like `!x_done` that is
       trivially TRUE immediately after `x` was just triggered (because `x`
       obviously isn't done yet) — that guard is a no-op disguised as a real
       one. If the query/comment implies the intent was "wait until X is
       actually finished," the fix is almost always to FLIP the guard from
       `!x_done` to `x_done`, not to bolt on an additional condition.

    5. TIMING FIXES: a `A[] not (Loc && clock > K)` violation means Loc's
       invariant is missing or too loose. Fix it by tightening the location's
       invariant to `clock <= K` (or adding guard `clock <= K` on the
       outgoing edge). Do NOT just add a lower-bound guard (`clock >= K`) —
       that only changes the earliest exit time, not the latest, and does
       nothing to prevent overstaying.

    6. DO NOT OVER-FIX / DO NOT VACUOUSLY SATISFY. A guard that can never
       become true (`false`, or a condition unreachable given the rest of the
       model) will make verifyta report the query as satisfied by making the
       associated action unreachable — this is NOT an acceptable fix, even
       though it "passes." Before finalizing a fix, mentally re-check: is
       every location and channel in the model still reachable? Can the
       mission still reach its completion/terminal location along some
       execution? If your fix would make any other query in the model newly
       fail or become vacuous, discard it and find a more targeted fix.

    7. MINIMALITY: prefer flipping or strengthening ONE existing guard over
       adding new guards, new booleans, new locations, or new channels. Only
       introduce a new global boolean flag if no existing flag captures the
       needed condition. Never rename existing locations, templates, or
       channels, and never delete or alter the <queries> block.
""")


def build_prompt_no_trace(xml_source: str, failures: list[dict]) -> str:
    failure_section = ""
    for f in failures:
        failure_section += textwrap.dedent(f"""
            ─────────────────────────────────────────────────────
            FAILING QUERY {f['index']}: `{f['query']}`
            Description: {f['comment']}

            What this query means:
              - "A[] P"     = P must hold in EVERY reachable state (safety/invariant).
              - "E<> P"     = there EXISTS a path where P eventually holds (reachability).
              - "A<> P"     = P must eventually hold on EVERY path (liveness).
              - "not deadlock" = the model must never reach a global state with
                                 zero enabled discrete transitions on every path
                                 forward, including after arbitrary time delay.
              - "imply"     = logical implication (A imply B ≡ !A || B).

            This property is currently NOT SATISFIED somewhere in the model's
            reachable state space. No counterexample trace is provided — you
            must reason about the model's structure directly:

              a) Identify which template(s), location(s), guard(s), and shared
                 variable(s) are named in or relevant to this query.
              b) Reason step by step about the orderings of transitions the
                 model currently allows between the relevant processes, and
                 work out which ordering makes this property false (for `A[]`
                 invariants) or unreachable (for `E<>`/`A<>` properties).
              c) Check whether a guard that looks like it should prevent or
                 ensure the property is in fact a no-op at the moment its
                 transition can fire (see rule 4 in the primer) — do not add a
                 redundant second guard; fix the existing one if so.
            ─────────────────────────────────────────────────────
        """)

    return textwrap.dedent(f"""
        You are a UPPAAL timed-automata expert repairing a Network of Timed
        Automata (NTA) so that every currently-failing query becomes SATISFIED,
        without breaking any query that currently passes.

        {UPPAAL_PRIMER_NO_TRACE}

        ═══════════════════════════════════════════════════════════
        PROCESS (do this reasoning silently, then output only the result):

        STEP 1 — For each failing query below, reason step by step about the
                 model's structure (NO execution trace is provided — you must
                 work out the fault purely from the queries and the model
                 XML) to find the exact transition and guard at fault.

        STEP 2 — Decide the minimal correct fix per rules 4–7 of the primer:
                 usually flipping/strengthening ONE existing guard or
                 tightening ONE invariant. Prefer fixes that reuse existing
                 boolean flags over introducing new ones.

        STEP 3 — Before finalizing, sanity-check your fix against EVERY query
                 in the model (not just the failing ones) — a fix must not
                 turn a previously-passing query into a failing or vacuous one.

        STEP 4 — Output ONLY the complete corrected XML.
                 • Start with exactly: <?xml
                 • No markdown fences (no ```), no prose, no comments outside
                   the XML itself.
                 • Keep every existing location id, template name, channel
                   name, and query unchanged.
        ═══════════════════════════════════════════════════════════

        {failure_section}

        ═══ CURRENT MODEL XML ═══
        {xml_source}
        ═════════════════════════

        Now output the corrected XML (starting with <?xml):
    """).strip()


def main():
    setup()

    src_path = MODEL_FILE
    attempt  = 0

    failures, passing, _ = verify_model(src_path)
    if not failures:
        ok("All queries already PASS. Nothing to fix.")
        return

    while attempt < MAX_REPAIR_ATTEMPTS:
        attempt += 1
        header(f"Attempt {attempt}/{MAX_REPAIR_ATTEMPTS} (no-trace) — {src_path}")

        print(f"\n{C.BOLD}{C.RED}  ✗ {len(failures)} query/queries FAILING:{C.RESET}")
        for f in failures:
            print(f"  {C.RED}[{f['index']}]{C.RESET} {f['query']}")
        print(f"  {C.DIM}({len(passing)} passing: {passing}){C.RESET}")

        xml_source = Path(src_path).read_text()
        prompt     = build_prompt_no_trace(xml_source, failures)

        info(f"Sending to LLM (no-trace, attempt {attempt})…")
        raw_reply  = call_llm(prompt)
        fixed_xml  = extract_xml(raw_reply)

        if not fixed_xml.startswith("<?xml"):
            err("LLM did not return valid XML. Aborting.")
            break

        try:
            ET.fromstring(fixed_xml)
        except ET.ParseError as e:
            err(f"LLM output is not well-formed XML ({e}). Aborting.")
            break

        if attempt == 1:
            shutil.copy(MODEL_FILE, MODEL_FILE + ".bak")
            dim(f"Original backed up to {MODEL_FILE}.bak")

        fixed_path = Path(src_path).parent / f"fixed_warehouse_no_trace_attempt{attempt}.xml"
        fixed_path.write_text(fixed_xml)
        info(f"LLM fix saved to: {C.BOLD}{fixed_path.name}{C.RESET}")
        log_diff(src_path, str(fixed_path))

        new_failures, new_passing, _ = verify_model(str(fixed_path))

        regressions = check_regressions(passing, new_failures)
        if regressions:
            err(f"REJECTED: fixed target failures but broke previously-passing "
                f"quer{'y' if len(regressions)==1 else 'ies'} {regressions}.")
            warn("Discarding this attempt and retrying from same source.")
            continue

        if not new_failures:
            ok("All queries PASS. Model is correctly repaired.")
            info(f"Fixed model: {C.BOLD}{fixed_path.name}{C.RESET}")
            shutil.copy(str(fixed_path), FIXED_FILE)
            info(f"Also copied to: {C.BOLD}{FIXED_FILE}{C.RESET}")
            return

        warn(f"Attempt {attempt} improved things but "
             f"{len(new_failures)} quer{'y' if len(new_failures)==1 else 'ies'} "
             f"still failing — continuing.")
        src_path = str(fixed_path)
        failures, passing = new_failures, new_passing

    err(f"Could not fully repair the model in {MAX_REPAIR_ATTEMPTS} attempt(s).")
    dim(f"Last attempt saved as: {src_path}")


if __name__ == "__main__":
    main()
