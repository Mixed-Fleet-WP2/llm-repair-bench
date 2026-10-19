"""
llm_fixer.py

Verifies a UPPAAL model, collects all failing queries + their counter-example
traces, sends them to the OpenAI API together with the full model XML, and
asks the LLM to return a fixed XML where every query is satisfied.
The repaired model is saved and re-verified automatically.

Includes:
  - A detailed, UPPAAL-semantics-aware repair prompt (guard scope rules,
    channel-type gotchas, committed-location pitfalls, guard-direction bugs,
    timing-fix rules, anti-over-fitting rules).
  - A regression check: a repair attempt is rejected if it fixes the target
    failures but breaks a query that was previously passing.
  - A line-level diff logged for every attempt, so over-fixing / unexpected
    restructuring is visible immediately instead of having to be re-derived
    from a fresh verifyta run.
"""

import difflib
import os
import shutil
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pyuppaal
import requests
from pyuppaal import UModel
from openai import OpenAI
import anthropic

# ── Terminal colours (plain ANSI — no extra dependencies) ────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    MAGENTA= "\033[35m"
    CYAN   = "\033[36m"
    WHITE  = "\033[37m"

def ok(msg):    print(f"{C.BOLD}{C.GREEN}✓ {msg}{C.RESET}")
def err(msg):   print(f"{C.BOLD}{C.RED}✗ {msg}{C.RESET}")
def warn(msg):  print(f"{C.YELLOW}⚠ {msg}{C.RESET}")
def info(msg):  print(f"{C.CYAN}  {msg}{C.RESET}")
def dim(msg):   print(f"{C.DIM}  {msg}{C.RESET}")
def header(msg):print(f"\n{C.BOLD}{C.BLUE}{'═'*60}\n  {msg}\n{'═'*60}{C.RESET}")

# ── Configuration ────────────────────────────────────────────────────────────

# Path to UPPAAL's verifyta binary: set the VERIFYTA environment variable,
# or put verifyta on your PATH.
VERIFYTA   = os.environ.get("VERIFYTA", "verifyta")
MODEL_FILE = "faulty_warehouse.xml"
FIXED_FILE = "fixed_warehouse.xml"
MAX_REPAIR_ATTEMPTS = 3

# API keys (read from environment)
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
XAI_API_KEY       = os.environ.get("XAI_API_KEY",       "")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY",  "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY",    "")

# ── Model registry ───────────────────────────────────────────────────────────
# Maps display name → (provider, api_model_id)
# provider: openai | anthropic | xai | deepseek | google | ollama

MODELS = {
    "gpt-5.1":           ("openai",    "gpt-5.1"),
    "gpt-5-mini":        ("openai",    "gpt-5-mini"),
    "gpt-5-nano":        ("openai",    "gpt-5-nano"),
    "gpt-4o":            ("openai",    "gpt-4o"),
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5"),
    "claude-haiku-4-5":  ("anthropic", "claude-haiku-4-5"),
    "grok-code-fast":    ("xai",       "grok-3-mini-fast"),
    "grok-3-mini":       ("xai",       "grok-3-mini"),
    "deepseek-reasoner": ("deepseek",  "deepseek-v4-pro"),
    "deepseek-coder:33b":   ("ollama",    "deepseek-coder:33b"),
    "qwen3.5:9b":        ("ollama",    "qwen3.5:9b"),
    "gemini-3.6-flash":  ("google",    "gemini-3.6-flash"),
    "gemini-2.5-pro":    ("google",    "gemini-2.5-pro"),
}

LLM_MODEL = "gpt-5.1"   # default for standalone llm_fixer usage

# Provider base URLs (OpenAI-compatible)
_PROVIDER_BASE = {
    "xai":      "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "google":   "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama":   "http://localhost:11434/v1",
}
_PROVIDER_KEY = {
    "openai":    lambda: OPENAI_API_KEY,
    "xai":       lambda: XAI_API_KEY,
    "deepseek":  lambda: DEEPSEEK_API_KEY,
    "google":    lambda: GOOGLE_API_KEY,
    "ollama":    lambda: "ollama",   # local — no real key needed
}

# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_MSG = (
    "You are a UPPAAL timed-automata expert specializing in "
    "coordination and concurrency bugs in multi-process NTA "
    "models. You always trace counterexamples step by step "
    "before proposing a fix, you prefer the minimal correct "
    "guard/invariant change over restructuring, and you never "
    "propose a fix that makes an action unreachable just to "
    "satisfy a query. You output ONLY the corrected XML — no "
    "markdown fences, no explanations, no extra text."
)


def setup():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            f"{C.RED}OPENAI_API_KEY is not set.\n"
            f"Export it before running:  export OPENAI_API_KEY='sk-...'{C.RESET}"
        )
    pyuppaal.set_verifyta_path(VERIFYTA)

# Usage of the most recent call_llm() request, filled by the provider calls:
# served_model (the model the provider says answered), input_tokens,
# output_tokens (visible + reasoning), reasoning_tokens, cached_input_tokens.
# Fields a provider does not report are left out. Empty if the call failed.
LAST_CALL: dict = {}


def _record_call(**fields) -> None:
    LAST_CALL.clear()
    LAST_CALL.update({k: v for k, v in fields.items() if v is not None})


def _openai_compatible(prompt: str, provider: str, api_id: str) -> str:
    """Call any OpenAI-compatible endpoint (openai, xai, deepseek, google, ollama)."""
    key      = _PROVIDER_KEY[provider]()
    base_url = _PROVIDER_BASE.get(provider)   # None → use OpenAI default
    client   = OpenAI(api_key=key, base_url=base_url)
    kwargs   = dict(
        model=api_id,
        messages=[
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
    )
    try:
        resp = client.chat.completions.create(**kwargs, temperature=1)
    except Exception as e:
        if "temperature" not in str(e).lower():
            raise
        resp = client.chat.completions.create(**kwargs)
    usage = resp.usage
    details_out = getattr(usage, "completion_tokens_details", None)
    details_in  = getattr(usage, "prompt_tokens_details", None)
    prompt_t     = getattr(usage, "prompt_tokens", None)
    completion_t = getattr(usage, "completion_tokens", None)
    total_t      = getattr(usage, "total_tokens", None)
    reasoning_t  = getattr(details_out, "reasoning_tokens", None)
    # Providers differ in what completion_tokens covers: OpenAI and DeepSeek
    # include reasoning in it, xAI reports reasoning on top of it, and Gemini
    # reports its thinking only through total_tokens. total - prompt is every
    # generated token in all three cases; any excess over completion_tokens is
    # reasoning that completion_tokens left out (completion_tokens is then the
    # visible output only).
    output_t = completion_t
    if None not in (total_t, prompt_t, completion_t) and total_t - prompt_t > completion_t:
        output_t = total_t - prompt_t
        reasoning_t = output_t - completion_t
    _record_call(
        served_model=(resp.model or "").removeprefix("models/") or None,
        input_tokens=prompt_t,
        output_tokens=output_t,
        reasoning_tokens=reasoning_t,
        cached_input_tokens=getattr(details_in, "cached_tokens", None),
    )
    return resp.choices[0].message.content.strip()


def _anthropic_call(prompt: str, api_id: str) -> str:
    """Call Anthropic Claude models."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=api_id,
        max_tokens=8192,
        system=_SYSTEM_MSG,
        messages=[{"role": "user", "content": prompt}],
    )
    _record_call(
        served_model=msg.model,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        cached_input_tokens=getattr(msg.usage, "cache_read_input_tokens", None),
    )
    return msg.content[0].text.strip()


def _ollama_call(prompt: str, api_id: str) -> str:
    """
    Call a local Ollama model via its native /api/chat endpoint, with
    thinking explicitly disabled.

    Ollama's OpenAI-compatible endpoint (/v1/chat/completions) does NOT
    honor `think: false` for reasoning-capable models (verified empirically
    against qwen3.5:9b — the request field is silently ignored there).
    Only the native API respects it. Without this, a thinking-capable model
    can spend its entire generation on internal <think> reasoning for a long
    prompt and return empty final content, wasting minutes per attempt for
    nothing. `think` is harmless to pass for non-thinking models — it's
    simply ignored.
    """
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": api_id,
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_MSG},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=1800,
    )
    resp.raise_for_status()
    body = resp.json()
    _record_call(
        served_model=body.get("model"),
        input_tokens=body.get("prompt_eval_count"),
        output_tokens=body.get("eval_count"),
    )
    return body["message"]["content"].strip()


def call_llm(prompt: str, model: str | None = None) -> str:
    """Route the prompt to the correct provider based on model name."""
    LAST_CALL.clear()
    model = model or LLM_MODEL
    provider, api_id = MODELS.get(model, ("openai", model))   # fallback: treat as openai id
    if provider == "anthropic":
        return _anthropic_call(prompt, api_id)
    if provider == "ollama":
        return _ollama_call(prompt, api_id)
    return _openai_compatible(prompt, provider, api_id)


def add_missing_layout(xml_path: str) -> None:
    """
    Give every <label> and <name> that lacks x/y drawing coordinates a default
    position, in place. The coordinates only place the text in the UPPAAL
    editor and do not affect semantics, but pyuppaal crashes on a label
    without them ("int() argument must be ... not 'NoneType'"), which would
    otherwise reject a valid LLM repair that simply omitted the layout.
    """
    text = Path(xml_path).read_text()
    root = ET.fromstring(text)
    missing = [e for tag in ("label", "name") for e in root.iter(tag)
               if e.get("x") is None or e.get("y") is None]
    if not missing:
        return
    for e in missing:
        e.set("x", e.get("x") or "0")
        e.set("y", e.get("y") or "0")
    decl = text[: text.find("?>") + 2] + "\n" if text.startswith("<?xml") else ""
    Path(xml_path).write_text(decl + ET.tostring(root, encoding="unicode"))


def verify_model(xml_path: str) -> tuple[list[dict], list[int], str]:
    """
    Returns (failures, passing_indices, raw_output).
    Each failure dict has keys: index, query, comment, trace.
    passing_indices is a list of 1-based query indices that are satisfied,
    used later to detect regressions introduced by a "fix".
    """
    add_missing_layout(xml_path)
    model = UModel(xml_path)
    raw = model.verify()

    failures = []
    passing  = []
    queries  = model.queries
    # parse comments from raw XML string
    root     = ET.fromstring(model.xml)
    comments = [q.findtext("comment", default="") for q in root.findall(".//query")]

    # Parse raw output per query block
    blocks = raw.strip().split("Verifying formula")
    for block in blocks[1:]:          # skip preamble
        lines = block.strip().splitlines()
        # find query index from "formula N at ..."
        idx = int(lines[0].split()[0]) - 1   # 1-based → 0-based
        satisfied = "-- Formula is satisfied." in block

        if satisfied:
            passing.append(idx + 1)
            continue

        # Grab the xtr trace file name from output
        xtr_path = None
        for line in lines:
            if "Writing witness trace to" in line:
                xtr_path = line.split("Writing witness trace to")[-1].strip()
                break

        trace_text = ""
        if xtr_path:
            try:
                trace = model.load_xtr_trace(xtr_path)
                trace_text = str(trace) if trace is not None else "(empty trace)"
            except Exception as e:
                trace_text = f"(could not parse trace: {e})"

        failures.append({
            "index":   idx + 1,
            "query":   queries[idx] if idx < len(queries) else "?",
            "comment": comments[idx] if idx < len(comments) else "",
            "trace":   trace_text,
        })

    return failures, passing, raw


# ── Prompt construction ──────────────────────────────────────────────────────

UPPAAL_PRIMER = textwrap.dedent("""
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

    4. GUARD-DIRECTION BUGS ARE THE #1 CAUSE OF FAILURES YOU WILL SEE. Before
       proposing any fix, trace through the counterexample and write out, step
       by step, what each relevant guard evaluates to at the moment its
       transition fires. Watch specifically for a guard like `!x_done` that is
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


def build_prompt(xml_source: str, failures: list[dict]) -> str:
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

            The counterexample trace below is a concrete execution that VIOLATES
            this query right now. Walk through it transition by transition:

              a) Identify the exact global state (combination of each process's
                 current location + variable values) where the query first
                 becomes false.
              b) Identify which transition, if it had NOT been allowed to fire
                 (or had fired later), would have prevented that state.
              c) Check what guard (if any) currently exists on that transition,
                 and evaluate it AT THE MOMENT it fires in this trace. If a
                 guard is already present but evaluates to true anyway, that is
                 almost certainly the bug (see rule 4 in the primer above) —
                 do not add a redundant second guard; fix the existing one.

            Counterexample trace:
            ```
            {f['trace'] or '(not available)'}
            ```
            ─────────────────────────────────────────────────────
        """)

    return textwrap.dedent(f"""
        You are a UPPAAL timed-automata expert repairing a Network of Timed
        Automata (NTA) so that every currently-failing query becomes SATISFIED,
        without breaking any query that currently passes.

        {UPPAAL_PRIMER}

        ═══════════════════════════════════════════════════════════
        PROCESS (do this reasoning silently, then output only the result):

        STEP 1 — For each failing query below, apply the trace-analysis method
                 from the primer to find the exact transition and guard at
                 fault.

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


def extract_xml(text: str) -> str:
    """Strip any accidental markdown fences the LLM may have added."""
    if "```" in text:
        lines = text.splitlines()
        inside = False
        xml_lines = []
        for line in lines:
            if line.startswith("```"):
                inside = not inside
                continue
            if inside or (not inside and xml_lines):
                xml_lines.append(line)
        text = "\n".join(xml_lines)
    # ensure we start at the XML declaration
    start = text.find("<?xml")
    if start > 0:
        text = text[start:]
    return text.strip()


def log_diff(old_path: str, new_path: str) -> None:
    """Print a unified line-level diff between two model files."""
    old_lines = Path(old_path).read_text().splitlines(keepends=True)
    new_lines = Path(new_path).read_text().splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=old_path, tofile=new_path, n=1,
    ))
    if not diff:
        warn("no textual changes detected — LLM returned an identical file")
        return
    info(f"Diff  {old_path}  →  {new_path}:")
    for line in diff:
        l = line.rstrip("\n")
        if l.startswith("+") and not l.startswith("+++"):
            print(f"   {C.GREEN}{l}{C.RESET}")
        elif l.startswith("-") and not l.startswith("---"):
            print(f"   {C.RED}{l}{C.RESET}")
        elif l.startswith("@@"):
            print(f"   {C.CYAN}{l}{C.RESET}")
        else:
            print(f"   {C.DIM}{l}{C.RESET}")


def check_regressions(passing_before: list[int], failures_after: list[dict]) -> list[int]:
    """
    Returns the list of query indices that were passing before the fix but
    are present in the failures list after the fix — i.e. queries the LLM's
    edit broke.
    """
    failed_after_indices = {f["index"] for f in failures_after}
    return sorted(i for i in passing_before if i in failed_after_indices)


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
        header(f"Attempt {attempt}/{MAX_REPAIR_ATTEMPTS}  —  {src_path}")

        print(f"\n{C.BOLD}{C.RED}  ✗ {len(failures)} query/queries FAILING:{C.RESET}")
        for f in failures:
            print(f"  {C.RED}[{f['index']}]{C.RESET} {f['query']}")
        print(f"  {C.DIM}({len(passing)} passing: {passing}){C.RESET}")

        xml_source = Path(src_path).read_text()
        prompt     = build_prompt(xml_source, failures)

        info(f"Sending to {C.BOLD}{LLM_MODEL}{C.RESET}{C.CYAN} (attempt {attempt})…")
        raw_reply  = call_llm(prompt)
        fixed_xml  = extract_xml(raw_reply)

        if not fixed_xml.startswith("<?xml"):
            err("LLM did not return valid XML. Aborting.")
            break

        # Validate it's at least well-formed XML before writing/verifying it
        try:
            ET.fromstring(fixed_xml)
        except ET.ParseError as e:
            err(f"LLM output is not well-formed XML ({e}). Aborting.")
            break

        # Back up original on first attempt
        if attempt == 1:
            shutil.copy(MODEL_FILE, MODEL_FILE + ".bak")
            dim(f"Original backed up to {MODEL_FILE}.bak")

        fixed_path = Path(src_path).parent / f"fixed_warehouse_attempt{attempt}.xml"
        fixed_path.write_text(fixed_xml)
        info(f"LLM fix saved to: {C.BOLD}{fixed_path.name}{C.RESET}")
        log_diff(src_path, str(fixed_path))

        # Re-verify the proposed fix
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