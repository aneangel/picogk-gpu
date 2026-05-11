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
from picogkgpu.sparse_reinit import SparseReinit  # noqa: E402


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


def bench_sparse(sizes: list[int], band_voxels: int = 5,
                 steps: int = 5, repeats: int = 3) -> list[dict]:
    """Sparse (NanoVDB) vs dense reinit on the same narrow-band sphere.

    Reports throughput as MCU/s where the cell count for sparse is the
    *active* voxel count, vs dense which always processes the full grid.
    """
    wp.init()
    device = "cuda" if wp.get_device().is_cuda else "cpu"
    results = []

    print()
    print(f"{'impl':<12}{'n':>6}{'band':>6}{'active':>10}{'frac':>7}{'wall(s)':>10}{'MCU/s':>10}")
    print("-" * 64)

    for n in sizes:
        dx = 1.0 / (n - 1)
        phi0 = _sphere_sdf(n, dx, radius=0.4)
        band = band_voxels * dx

        # dense baseline (processes every cell)
        dense_op = Reinit((n, n, n), dx=dx, device=device)
        dense_op.run(phi0, num_steps=steps)
        t_dense = _time(lambda: dense_op.run(phi0, num_steps=steps), repeats=repeats)
        dense_mcus = (n ** 3 * steps) / t_dense / 1e6
        results.append({"impl": f"dense:{device}", "n": n, "wall_s": t_dense,
                        "mcus": dense_mcus, "cells": n ** 3, "band_voxels": band_voxels})
        print(f"{'dense':<12}{n:>6}{band_voxels:>6}{n**3:>10,}{1.0:>7.2%}{t_dense:>10.3f}{dense_mcus:>10.1f}")

        # sparse: only active band cells are processed
        sparse_op = SparseReinit(phi0, dx=dx, band_width=band, device=device)
        sparse_op.run(num_steps=steps)
        t_sparse = _time(lambda: sparse_op.run(num_steps=steps), repeats=repeats)
        sparse_mcus = (sparse_op.n_active * steps) / t_sparse / 1e6
        frac = sparse_op.n_active / (n ** 3)
        results.append({"impl": f"sparse:{device}", "n": n, "wall_s": t_sparse,
                        "mcus": sparse_mcus, "cells": sparse_op.n_active,
                        "active_fraction": frac, "band_voxels": band_voxels})
        print(f"{'sparse':<12}{n:>6}{band_voxels:>6}{sparse_op.n_active:>10,}{frac:>7.2%}{t_sparse:>10.3f}{sparse_mcus:>10.1f}")

    return results


def main():
    # numpy ref is O(n^3) per step with constant overhead; keep grids small.
    # warp scales up; 512^3 ~= 134M cells, comfortably under 32GB VRAM.
    result = bench(
        sizes_numpy=[32, 48, 64],
        sizes_warp=[64, 128, 256, 512],
        steps=5,
        repeats=3,
    )
    result["results"].extend(bench_sparse(sizes=[128, 256, 512], band_voxels=5, steps=5, repeats=3))

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_reinit_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
