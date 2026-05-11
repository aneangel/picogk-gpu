"""Microbenchmark: HJ-WENO5 reinit, Warp GPU vs numpy reference.

Measures wall time per reinit step at several grid sizes and reports the
throughput in million-cell-updates per second (MCU/s) so results can be
compared across grid sizes and (eventually) to OpenVDB's reinit numbers from
the perf profile.

Usage:
  uv run python benchmarks/micro/bench_reinit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from picogkgpu.reinit import Reinit, reinit_numpy, reinit_warp  # noqa: E402


def _sphere_sdf(n: int, dx: float, radius: float) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) - radius).astype(np.float32)


def _time(fn, repeats: int = 3) -> float:
    """Median of `repeats` wall-time samples in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench(sizes_numpy: list[int], sizes_warp: list[int], steps: int = 5, repeats: int = 3) -> dict:
    wp.init()
    device = "cuda" if wp.get_device().is_cuda else "cpu"

    out: dict = {"device": device, "steps": steps, "repeats": repeats, "results": []}

    print(f"{'impl':<10}{'n':>6}{'steps':>7}{'wall(s)':>10}{'MCU/s':>12}{'cells':>14}")
    print("-" * 60)

    for n in sizes_numpy:
        dx = 1.0 / (n - 1)
        phi0 = _sphere_sdf(n, dx, radius=0.4)
        # warm up
        reinit_numpy(phi0, dx=dx, num_steps=1)
        t = _time(lambda: reinit_numpy(phi0, dx=dx, num_steps=steps), repeats=repeats)
        cells = n ** 3
        mcus = (cells * steps) / t / 1e6
        out["results"].append({"impl": "numpy", "n": n, "wall_s": t, "mcus": mcus, "cells": cells})
        print(f"{'numpy':<10}{n:>6}{steps:>7}{t:>10.3f}{mcus:>12.2f}{cells:>14,}")

    for n in sizes_warp:
        dx = 1.0 / (n - 1)
        phi0 = _sphere_sdf(n, dx, radius=0.4)
        cells = n ** 3

        # variant A: one-shot helper, re-allocates and re-captures every call
        reinit_warp(phi0, dx=dx, num_steps=1, device=device)
        t_loop = _time(lambda: reinit_warp(phi0, dx=dx, num_steps=steps, device=device), repeats=repeats)
        mcus_loop = (cells * steps) / t_loop / 1e6
        out["results"].append({"impl": f"warp-oneshot:{device}", "n": n, "wall_s": t_loop, "mcus": mcus_loop, "cells": cells})
        print(f"{'warp/1shot':<10}{n:>6}{steps:>7}{t_loop:>10.3f}{mcus_loop:>12.2f}{cells:>14,}")

        # variant B: Reinit class — allocates and captures once, reuses graph
        op = Reinit((n, n, n), dx=dx, device=device)
        op.run(phi0, num_steps=steps)  # warm: triggers graph build
        t_graph = _time(lambda: op.run(phi0, num_steps=steps), repeats=repeats)
        mcus_graph = (cells * steps) / t_graph / 1e6
        out["results"].append({"impl": f"warp-graph:{device}", "n": n, "wall_s": t_graph, "mcus": mcus_graph, "cells": cells})
        print(f"{'warp/graph':<10}{n:>6}{steps:>7}{t_graph:>10.3f}{mcus_graph:>12.2f}{cells:>14,}")

    return out


def main():
    # numpy ref is O(n^3) per step with constant overhead; keep grids small.
    # warp scales up; 512^3 ~= 134M cells, comfortably under 32GB VRAM.
    result = bench(
        sizes_numpy=[32, 48, 64],
        sizes_warp=[64, 128, 256, 512],
        steps=5,
        repeats=3,
    )

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_reinit_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
