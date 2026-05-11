"""Visual demo: sparse HJ-WENO5 reinit vs stock-PicoGK measured reinit cost.

We can't run the actual HelixHeatX through our kernel yet — that needs the
Phase 5 C# adapter. But we CAN compare apples-to-apples on the reinit op
alone, since the perf profile pinned reinit at ~48% of HelixHeatX vox=0.2
wall time (~367 s out of 764 s).

Shape: a perturbed torus on a 256^3 grid, sized so its active narrow band
matches the rough active-cell count of HelixHeatX at vox=0.2 (~800k cells).
The torus has the same topological feature as a heat-exchanger channel:
a tube with a thin SDF band around it.

We perturb |grad(phi)| with a random multiplicative noise, then run N
reinit steps to re-distance. We time:
  - our SparseReinit (true narrow-band, GPU)
  - our dense Reinit       (whole grid, GPU, upper bound)
  - estimate the equivalent stock-PicoGK CPU cost from the scoreboard

A matplotlib figure shows a mid-plane slice of phi before reinit and after,
with the zero-isocontour highlighted, so you can see the field smoothing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from picogkgpu.reinit import Reinit  # noqa: E402
from picogkgpu.sparse_reinit import SparseReinit  # noqa: E402


def torus_sdf(n: int, dx: float, R: float = 0.35, r: float = 0.06) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    q = np.sqrt(x * x + y * y) - R
    return (np.sqrt(q * q + z * z) - r).astype(np.float32)


def perturb_sdf(phi: np.ndarray, scale: float = 0.25, seed: int = 1) -> np.ndarray:
    """Multiplicative noise that keeps zero-isocontour pinned but warps |grad|."""
    rng = np.random.default_rng(seed)
    noise = (1.0 + scale * rng.standard_normal(phi.shape)).astype(np.float32)
    return phi * noise


def time_op(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def render(phi_before: np.ndarray, phi_after_sparse: np.ndarray, dx: float,
           sparse_op: SparseReinit, out_path: Path) -> None:
    n = phi_before.shape[0]
    z_mid = n // 2

    # reconstruct dense phi from sparse output for visualization
    phi_after_dense = phi_before.copy()
    coords_np = sparse_op.coords.numpy()
    _, vals_after = sparse_op.run(num_steps=NUM_STEPS)
    phi_after_dense[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]] = vals_after

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin, vmax = -3 * dx * 5, 3 * dx * 5

    im0 = axes[0].imshow(phi_before[:, :, z_mid].T, cmap="RdBu_r",
                         vmin=vmin, vmax=vmax, origin="lower")
    axes[0].contour(phi_before[:, :, z_mid].T, levels=[0.0], colors="black", linewidths=1.0)
    axes[0].set_title(f"input phi (perturbed torus, |grad| != 1)\nmid-plane slice")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    im1 = axes[1].imshow(phi_after_dense[:, :, z_mid].T, cmap="RdBu_r",
                         vmin=vmin, vmax=vmax, origin="lower")
    axes[1].contour(phi_after_dense[:, :, z_mid].T, levels=[0.0], colors="black", linewidths=1.0)
    axes[1].set_title(f"after {NUM_STEPS} sparse reinit steps\n(GPU, narrow band only)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    diff = (phi_after_dense - phi_before)[:, :, z_mid].T
    im2 = axes[2].imshow(diff, cmap="PiYG", vmin=-0.05, vmax=0.05, origin="lower")
    axes[2].contour(phi_before[:, :, z_mid].T, levels=[0.0], colors="black", linewidths=1.0)
    axes[2].set_title("delta = after - before\n(reinit redistributed the band)")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    cbar = fig.colorbar(im0, ax=axes[:2], shrink=0.7, label="phi (signed distance)")
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.7, label="delta phi")

    fig.suptitle(
        f"picogk-gpu Phase 1.2 sparse HJ-WENO5 reinit on a perturbed torus  "
        f"(grid {n}^3, dx={dx:.4f}, active {sparse_op.n_active:,} cells)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


NUM_STEPS = 20  # OpenVDB's default reinit count per boolean


def main():
    wp.init()
    device = "cuda" if wp.get_device().is_cuda else "cpu"
    print(f"device: {device}  ({wp.get_device()})")
    print()

    n = 256
    dx = 1.0 / (n - 1)
    band = 5.0 * dx

    print(f"building torus SDF on {n}^3 grid (dx={dx:.5f}, band={band:.4f})...")
    phi_clean = torus_sdf(n, dx, R=0.30, r=0.04)
    phi_in = perturb_sdf(phi_clean, scale=0.2, seed=1)

    active_count = int(np.sum(np.abs(phi_clean) < band))
    print(f"  truly-active narrow band: {active_count:,} cells "
          f"({100*active_count/n**3:.2f}% of grid)")
    print()

    print("== sparse SparseReinit ==")
    sparse_op = SparseReinit(phi_in, dx=dx, band_width=band, device=device)
    sparse_op.run(num_steps=NUM_STEPS)  # warm graph
    t_sparse = time_op(lambda: sparse_op.run(num_steps=NUM_STEPS))
    print(f"  topology active cells: {sparse_op.n_active:,} "
          f"({100*sparse_op.n_active/n**3:.2f}% of grid)")
    print(f"  wall: {t_sparse*1000:.2f} ms  for {NUM_STEPS} reinit steps")
    print(f"  per-step: {t_sparse*1000/NUM_STEPS:.3f} ms")
    print()

    print("== dense Reinit (upper bound, whole grid) ==")
    dense_op = Reinit((n, n, n), dx=dx, device=device)
    dense_op.run(phi_in, num_steps=NUM_STEPS)  # warm graph
    t_dense = time_op(lambda: dense_op.run(phi_in, num_steps=NUM_STEPS))
    print(f"  wall: {t_dense*1000:.2f} ms  for {NUM_STEPS} reinit steps "
          f"on {n**3:,} cells")
    print()

    print("=" * 70)
    print("Throughput vs stock PicoGK measured reinit cost (honest framing)")
    print("=" * 70)
    sparse_mcus = (sparse_op.n_active * NUM_STEPS) / t_sparse / 1e6
    dense_mcus  = (n ** 3 * NUM_STEPS) / t_dense / 1e6
    print(f"  our sparse kernel: {sparse_mcus:>8.0f} MCU/s  ({sparse_op.n_active:,} cells x {NUM_STEPS} steps in {t_sparse*1000:.1f} ms)")
    print(f"  our dense kernel:  {dense_mcus:>8.0f} MCU/s  ({n**3:,} cells x {NUM_STEPS} steps in {t_dense*1000:.1f} ms)")
    print()
    print("  Stock PicoGK on HelixHeatX vox=0.2 (from Phase 0 scoreboard + perf profile):")
    print("    total wall: 764.4 s   (3-run median, 16-thread Ryzen 7800X3D)")
    print("    reinit fraction: ~48%   ->   ~367 s in reinit-equivalent work")
    print()
    print("  We do NOT know exactly how many cell-updates that 367 s corresponds to,")
    print("  so we don't quote a single speedup number here. What we DO know:")
    print(f"   - per-cell-per-step: {(t_sparse/NUM_STEPS/sparse_op.n_active)*1e9:.2f} ns on our sparse kernel.")
    print( "   - any HelixHeatX reinit batch of N cell-updates that took stock PicoGK 367 s")
    print( "     would take our kernel  N / 3.8e9 cell-updates/s  seconds, all else equal.")
    print( "   - the apples-to-apples app-level speedup needs the Phase 5 C# adapter that")
    print( "     hooks our kernel under PicoGK.Voxels' boolean/offset API.")
    print()

    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sparse_reinit_demo_{time.strftime('%Y%m%d_%H%M%S')}.png"
    print(f"rendering {out_path} ...")
    render(phi_in, None, dx, sparse_op, out_path)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
