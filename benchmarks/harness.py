"""Benchmark runner: invokes a dotnet project and captures wall time + peak RSS.

Two-step run model:
  1. `dotnet build -c Release` once, UNTIMED — first invocation is otherwise
     dominated by compile time, which has nothing to do with the engine.
  2. `dotnet run -c Release --no-build` N times under /usr/bin/time -v — TIMED.

Output: one JSON per invocation in `results/`, plus aggregate stats across runs.

Usage:
  python benchmarks/harness.py \\
      --name helix_heatx \\
      --project benchmarks/helix_heatx/Host \\
      --voxel-size 0.5 \\
      --runs 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from env import capture as capture_env  # noqa: E402

TIME_BIN = "/usr/bin/time"


def _parse_gnu_time(stderr: str) -> dict:
    out: dict = {}
    patterns = {
        "wall_seconds": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): (\d+)",
        "user_seconds": r"User time \(seconds\): ([\d.]+)",
        "system_seconds": r"System time \(seconds\): ([\d.]+)",
        "cpu_percent": r"Percent of CPU this job got: (\d+)%",
        "page_faults_major": r"Major \(requiring I/O\) page faults: (\d+)",
        "page_faults_minor": r"Minor \(reclaiming a frame\) page faults: (\d+)",
        "context_switches_vol": r"Voluntary context switches: (\d+)",
        "context_switches_invol": r"Involuntary context switches: (\d+)",
        "exit_status": r"Exit status: (\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stderr)
        if not m:
            continue
        val = m.group(1)
        if key == "wall_seconds":
            parts = val.split(":")
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            out[key] = secs
        elif "." in val:
            out[key] = float(val)
        else:
            out[key] = int(val)
    return out


def _check_prereqs() -> None:
    missing = []
    if not Path(TIME_BIN).exists():
        missing.append(f"{TIME_BIN} (install: apt install time)")
    if not shutil.which("dotnet"):
        missing.append("dotnet (install .NET 9 SDK: https://learn.microsoft.com/dotnet/core/install/linux)")
    if missing:
        sys.stderr.write("missing prerequisites:\n")
        for m in missing:
            sys.stderr.write(f"  - {m}\n")
        sys.exit(2)


def _run_under_time(cmd: list[str], cwd: Path, env: dict) -> dict:
    wrapped = [TIME_BIN, "-v", *cmd]
    proc = subprocess.run(
        wrapped, cwd=str(cwd), env=env, capture_output=True, text=True,
    )
    metrics = _parse_gnu_time(proc.stderr)
    metrics["exit_code"] = proc.returncode
    metrics["stdout_tail"] = (proc.stdout or "")[-2000:]
    metrics["stderr_tail"] = (proc.stderr or "")[-2000:]
    return metrics


def _aggregate(runs: list[dict]) -> dict:
    fields = ("wall_seconds", "max_rss_kb", "user_seconds", "system_seconds", "cpu_percent")
    agg: dict = {}
    for f in fields:
        vals = [r[f] for r in runs if isinstance(r.get(f), (int, float))]
        if not vals:
            continue
        agg[f] = {
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "stddev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return agg


def run_benchmark(
    *,
    name: str,
    project_dir: Path,
    runs: int = 1,
    voxel_size: float | None = None,
    extra_args: list[str] | None = None,
    output_dir: Path,
) -> dict:
    _check_prereqs()
    project_dir = Path(project_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = list(extra_args or [])

    env = os.environ.copy()
    if voxel_size is not None:
        env["PICOGK_VOXEL_SIZE"] = str(voxel_size)

    print(f"[{name}] building {project_dir} ...", flush=True)
    build = subprocess.run(
        ["dotnet", "build", "-c", "Release", "--nologo", "-v", "quiet", str(project_dir)],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        sys.exit(build.returncode)

    cmd = [
        "dotnet", "run", "-c", "Release", "--no-build",
        "--project", str(project_dir), "--", *extra_args,
    ]
    print(f"[{name}] {runs} run(s) @ voxel={voxel_size}: {' '.join(cmd)}", flush=True)

    run_results = []
    for i in range(runs):
        print(f"  run {i + 1}/{runs} ...", flush=True)
        t0 = time.time()
        r = _run_under_time(cmd, cwd=project_dir, env=env)
        r["run_idx"] = i
        r["python_wall_seconds"] = time.time() - t0
        if r.get("exit_code") != 0:
            sys.stderr.write(f"[{name}] run {i} exited {r['exit_code']}\n")
            sys.stderr.write(r.get("stderr_tail", ""))
        run_results.append(r)

    record = {
        "name": name,
        "voxel_size": voxel_size,
        "args": extra_args,
        "runs": run_results,
        "aggregate": _aggregate(run_results),
        "env": capture_env(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vox_str = f"_vox{voxel_size}" if voxel_size is not None else ""
    out_path = output_dir / f"{name}{vox_str}_{timestamp}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  -> {out_path}", flush=True)
    return record


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="benchmark label (used in output filename)")
    parser.add_argument("--project", required=True, help="path to host project directory")
    parser.add_argument("--voxel-size", type=float, help="voxel size in mm, exported as PICOGK_VOXEL_SIZE")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--arg", action="append", default=[],
                        help="extra arg passed after `--` to dotnet run (repeatable)")
    a = parser.parse_args()
    run_benchmark(
        name=a.name,
        project_dir=Path(a.project),
        runs=a.runs,
        voxel_size=a.voxel_size,
        extra_args=a.arg,
        output_dir=Path(a.output_dir),
    )


if __name__ == "__main__":
    _cli()
