"""
check_api_keys.py

Pre-flight check before a benchmark run. For every model in llm_fixer.MODELS
(or the ones given with --models) it checks:

  1. KEY    the provider's API key env var is set
  2. AUTH   the provider accepts the key (lists its models — free, no tokens)
  3. MODEL  the api model id is available on that provider, so a deprecated
            endpoint or a missing Ollama tag is caught up front instead of
            producing a 0% run of HTTP 404s. Ids missing from the provider's
            model list (aliases such as `claude-haiku-4-5`, which is listed
            only as `claude-haiku-4-5-20251001`) are confirmed with one tiny
            real request instead of being rejected.
  4. SERVED the model that actually answers is the one requested. Some
            providers silently serve a different model for a retired id
            (xAI answers `grok-3-mini` with `grok-4.3`), which would put
            another model's results under the wrong name. A dated version
            of the requested id (`gpt-4o-2024-08-06`) counts as a match.

--ping sends the tiny request for every model, not only unlisted ones, so the
SERVED check covers all of them (costs a few tokens per model).

Usage (from the repo root):
    python goal_reachability/check_api_keys.py
    python goal_reachability/check_api_keys.py --models gpt-4o,claude-haiku-4-5 --ping
"""

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import anthropic                                           # noqa: E402
from openai import OpenAI                                  # noqa: E402
from llm_fixer import (MODELS, ANTHROPIC_API_KEY, C,       # noqa: E402
                       _PROVIDER_BASE, _PROVIDER_KEY)

ENV_VAR = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai":       "XAI_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "ollama":    None,
}


def list_provider_models(provider: str) -> set[str]:
    """Model ids the provider exposes for this key. Raises on auth failure."""
    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return {m.id for m in client.models.list(limit=1000)}
    if provider == "ollama":
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        return {m["name"] for m in r.json()["models"]}
    client = OpenAI(api_key=_PROVIDER_KEY[provider](), base_url=_PROVIDER_BASE.get(provider))
    # Google's OpenAI-compatible endpoint prefixes ids with "models/".
    return {m.id.removeprefix("models/") for m in client.models.list()}


def served_model(provider: str, api_id: str) -> str:
    """Send one tiny request and return the id of the model that answered."""
    prompt = [{"role": "user", "content": "Reply with the single word OK."}]
    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return client.messages.create(model=api_id, max_tokens=5, messages=prompt).model
    if provider == "ollama":
        r = requests.post("http://localhost:11434/api/chat", timeout=600,
                          json={"model": api_id, "messages": prompt,
                                "stream": False, "think": False})
        r.raise_for_status()
        return r.json().get("model", api_id)
    client = OpenAI(api_key=_PROVIDER_KEY[provider](), base_url=_PROVIDER_BASE.get(provider))
    reply = client.chat.completions.create(model=api_id, messages=prompt)
    return (reply.model or api_id).removeprefix("models/")


def main():
    ap = argparse.ArgumentParser(description="Pre-flight API key and model check.")
    ap.add_argument("--models", default="all",
                    help=f"Comma-separated names or 'all'. Available: {', '.join(MODELS)}")
    ap.add_argument("--ping", action="store_true",
                    help="Also send one tiny chat request per model (costs a few tokens).")
    args = ap.parse_args()

    names = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}")

    cache: dict[str, set[str] | Exception] = {}
    ok_models, bad = [], 0
    for name in names:
        provider, api_id = MODELS[name]
        env = ENV_VAR[provider]
        label = f"{name:<20} {provider:<9} {api_id:<22}"

        if env and not os.environ.get(env):
            print(f"{C.RED}✗ {label} KEY   {env} is not set{C.RESET}")
            bad += 1
            continue

        if provider not in cache:
            try:
                cache[provider] = list_provider_models(provider)
            except Exception as e:
                cache[provider] = e
        available = cache[provider]
        if isinstance(available, Exception):
            what = "Ollama not reachable" if provider == "ollama" else "key rejected / API error"
            print(f"{C.RED}✗ {label} AUTH  {what}: {str(available)[:120]}{C.RESET}")
            bad += 1
            continue

        if provider == "ollama" and api_id not in available:
            print(f"{C.RED}✗ {label} MODEL {api_id} not pulled (run: ollama pull {api_id}){C.RESET}")
            bad += 1
            continue

        served = api_id
        if args.ping or api_id not in available:
            try:
                served = served_model(provider, api_id)
            except Exception as e:
                print(f"{C.RED}✗ {label} MODEL {api_id} not available: {str(e)[:120]}{C.RESET}")
                bad += 1
                continue
            if not served.startswith(api_id):
                print(f"{C.RED}✗ {label} SERVED provider answers with {served} instead — "
                      f"results would be mislabelled{C.RESET}")
                bad += 1
                continue

        extra = f" (served as {served})" if served != api_id else ""
        print(f"{C.GREEN}✓ {label} ready{extra}{C.RESET}")
        ok_models.append(name)

    print(f"\n{len(ok_models)}/{len(names)} models ready.")
    if ok_models:
        print(f"Run only these with:\n  --models {','.join(ok_models)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
