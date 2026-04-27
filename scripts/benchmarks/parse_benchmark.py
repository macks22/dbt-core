#!/usr/bin/env python3
"""Benchmark dbt parse cold-cache wall time and perf_info breakdown.

Runs ``dbt parse --no-partial-parse`` against an arbitrary dbt project N times
with a clean ``target/`` directory each time, then writes a one-line summary
(median wall, parse_project_elapsed, etc.) to a CSV.

Use ``--project-id`` to anonymize project names in the committed CSV
(``small``/``medium``/``large``) so benchmark results can be shared without
leaking project-internal identifiers.

Usage:
    python scripts/benchmarks/parse_benchmark.py \\
        --project /path/to/dbt/project \\
        --project-id small \\
        --label baseline \\
        --repeats 3 \\
        --out scripts/benchmarks/results/perf.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(project_dir: Path) -> dict:
    target = project_dir / "target"
    if target.exists():
        shutil.rmtree(target)
    from dbt.cli.main import dbtRunner

    t0 = time.monotonic()
    result = dbtRunner().invoke([
        "parse",
        "--no-partial-parse",
        "--project-dir",
        str(project_dir),
    ])
    elapsed = time.monotonic() - t0
    perf_path = target / "perf_info.json"
    perf = json.loads(perf_path.read_text()) if perf_path.exists() else {}
    return {
        "wall_s": elapsed,
        "load_all_elapsed": perf.get("load_all_elapsed"),
        "parse_project_elapsed": perf.get("parse_project_elapsed"),
        "load_macros_elapsed": perf.get("load_macros_elapsed"),
        "process_manifest_elapsed": perf.get("process_manifest_elapsed"),
        "path_count": perf.get("path_count"),
        "success": bool(result.success),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Path to dbt project (containing dbt_project.yml)")
    ap.add_argument("--label", required=True, help="Identifier for this run (e.g., 'baseline', 'phase1')")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None, help="Optional CSV append path")
    ap.add_argument(
        "--project-id",
        default=None,
        help="Anonymized project name for results CSV (e.g., 'small', 'medium', 'large')",
    )
    args = ap.parse_args()

    project_dir = Path(args.project).resolve()
    if not (project_dir / "dbt_project.yml").exists():
        print(f"error: {project_dir}/dbt_project.yml not found", file=sys.stderr)
        return 2

    samples = []
    for i in range(args.repeats):
        print(f"[run {i + 1}/{args.repeats}] cold parse {project_dir}", flush=True)
        samples.append(run_once(project_dir))

    keys = (
        "wall_s",
        "load_all_elapsed",
        "parse_project_elapsed",
        "load_macros_elapsed",
        "process_manifest_elapsed",
    )
    medians = {k: statistics.median(s[k] for s in samples if s[k] is not None) for k in keys}
    stdev = statistics.stdev(s["wall_s"] for s in samples) if args.repeats > 1 else 0.0
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parent,
    ).strip()

    summary = {
        "label": args.label,
        "project_id": args.project_id or project_dir.name,
        "path_count": samples[0].get("path_count"),
        "wall_s_median": round(medians["wall_s"], 2),
        "wall_s_stdev": round(stdev, 2),
        "parse_project_elapsed_median": round(medians["parse_project_elapsed"], 2),
        "load_all_elapsed_median": round(medians["load_all_elapsed"], 2),
        "load_macros_elapsed_median": round(medians["load_macros_elapsed"], 2),
        "process_manifest_elapsed_median": round(medians["process_manifest_elapsed"], 2),
        "dbt_core_sha": sha,
        "repeats": args.repeats,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                w.writeheader()
            w.writerow(summary)
    return 0 if all(s["success"] for s in samples) else 1


if __name__ == "__main__":
    sys.exit(main())
