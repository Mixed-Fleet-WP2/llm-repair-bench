"""
make_goal_dataset.py

Builds dataset/: a copy of the missions/ dataset where each
mission also carries goal-reachability queries, so a repair that satisfies the
original properties vacuously (e.g. by time-locking the model) can be detected.

For every mission:
  - goal locations = the sink locations (no outgoing edge) of each process
    instance in faulty.xml, e.g. `crane.Done`
  - goal queries   = `E<> p.Goal` per process. With --joint, also one joint
                     query `E<> (p.Goal && q.Goal)`; off by default because it
                     rejects correct repairs that add a location after a
                     robot's goal (the robot finishes, then moves on)
  - every goal query is verified with verifyta on faulty.xml and kept only if
    it is satisfied there — the goal must be reachable before repair, so a
    repair that makes it unreachable has broken the mission

Output per mission (dataset/<NN_name>/):
  faulty.xml        original model with the goal queries appended to <queries>
  queries.q         original queries followed by the goal queries
  goal_queries.q    goal queries only — run against any repaired model:
                      verifyta -q repaired.xml goal_queries.q
  metadata.json     original metadata plus a "goal_reachability" section

This script documents how the benchmark dataset under dataset/ was built from the
original 30-mission benchmark, which is not part of this package.

Usage:
    python goal_reachability/make_goal_dataset.py --src <original-missions-folder>
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from llm_fixer import VERIFYTA  # noqa: E402


def instances(xml: str) -> dict[str, str]:
    """Map process instance name -> template name from the <system> block."""
    sysb = re.search(r"<system>(.*?)</system>", xml, re.S).group(1)
    inst = {a: b for a, b in re.findall(r"(\w+)\s*=\s*(\w+)\s*\(", sysb)}
    listed = re.search(r"system\s+([\w\s,]+);", sysb).group(1)
    return {p.strip(): inst.get(p.strip(), p.strip()) for p in listed.split(",")}


def sink_locations(xml: str) -> dict[str, list[str]]:
    """Template name -> names of its locations that have no outgoing edge."""
    out = {}
    for t in re.findall(r"<template>(.*?)</template>", xml, re.S):
        name = re.search(r"<name[^>]*>(\w+)</name>", t).group(1)
        locs = dict(re.findall(r'<location id="(\w+)"[^>]*>\s*<name[^>]*>(\w+)</name>', t))
        sources = set(re.findall(r'<source ref="(\w+)"', t))
        out[name] = [loc for lid, loc in locs.items() if lid not in sources]
    return out


def verify(xml_path: Path, queries: list[str]) -> list[bool]:
    """Run verifyta on xml_path with the given queries; one bool per query."""
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as f:
        f.write("\n".join(queries) + "\n")
        qfile = f.name
    try:
        raw = subprocess.run([VERIFYTA, "-q", str(xml_path), qfile],
                             capture_output=True, text=True, timeout=300).stdout
    finally:
        Path(qfile).unlink()
    res = re.findall(r"Formula is (NOT )?satisfied", raw)
    if len(res) != len(queries):
        raise RuntimeError(f"verifyta returned {len(res)} results for "
                           f"{len(queries)} queries on {xml_path}:\n{raw}")
    return [r == "" for r in res]


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_mission(src: Path, dst: Path, joint: bool = False) -> dict:
    xml = (src / "faulty.xml").read_text()
    sinks = sink_locations(xml)

    goals = {p: [f"{p}.{loc}" for loc in sinks.get(t, [])]
             for p, t in instances(xml).items()}
    per_process = [(f"E<> {g}", f"Goal preserved: {p} can still reach {g.split('.')[1]}")
                   for p, gs in goals.items() for g in gs]
    candidates = list(per_process)
    # Joint goal only when each process has exactly one goal location.
    if joint and all(len(gs) == 1 for gs in goals.values()) and len(goals) > 1:
        joint = " && ".join(gs[0] for gs in goals.values())
        candidates.append((f"E<> ({joint})",
                           "Goal preserved: the whole mission can complete on one run"))

    ok = verify(src / "faulty.xml", [q for q, _ in candidates])
    kept     = [c for c, v in zip(candidates, ok) if v]
    rejected = [c[0] for c, v in zip(candidates, ok) if not v]

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("faulty.xml", "queries.q", "metadata.json") \
                and not f.name.startswith("faulty_xtr") and f.name != "llm_repaired.xml":
            shutil.copy(f, dst / f.name)

    goal_xml = "".join(f"<query><formula>{xml_escape(q)}</formula>"
                       f"<comment>{xml_escape(c)}</comment></query>" for q, c in kept)
    (dst / "faulty.xml").write_text(xml.replace("</queries>", goal_xml + "</queries>", 1))

    orig_q = [l.strip() for l in (src / "queries.q").read_text().splitlines() if l.strip()]
    (dst / "queries.q").write_text("\n".join(orig_q + [q for q, _ in kept]) + "\n")
    (dst / "goal_queries.q").write_text("\n".join(q for q, _ in kept) + "\n")

    meta = json.loads((src / "metadata.json").read_text())
    meta["goal_reachability"] = {
        "goal_locations": goals,
        "queries": [{"formula": q, "comment": c, "expected": "PASS"} for q, c in kept],
        "rejected_not_reachable_in_faulty": rejected,
    }
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Sanity check: the new faulty.xml must parse and keep the original
    # verdicts plus every goal query passing.
    verdicts = verify(dst / "faulty.xml", orig_q + [q for q, _ in kept])
    assert all(verdicts[len(orig_q):]), f"goal query failed on {dst}/faulty.xml"
    return {"mission": src.name, "kept": len(kept), "rejected": rejected}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True,
                    help="Folder with the original benchmark (faulty.xml, queries.q, "
                         "metadata.json per mission). Not part of this package; the "
                         "generated benchmark is included as dataset/.")
    ap.add_argument("--out", default=str(HERE.parent / "dataset"))
    ap.add_argument("--joint", action="store_true",
                    help="Also add the joint goal query E<> (a.Goal && b.Goal).")
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    for mdir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        r = build_mission(mdir, out_root / mdir.name, joint=args.joint)
        note = f"  rejected: {r['rejected']}" if r["rejected"] else ""
        print(f"{r['mission']:<34} {r['kept']} goal queries{note}")


if __name__ == "__main__":
    main()
