"""Visual demo: a "heat-exchanger-like" CSG construction using Phase 1.3 ops.

This is the smallest piece of CSG that has the same *topology* as a real
heat exchanger: a bounding solid with two parallel internal channels carved
out. Stock-PicoGK constructs HelixHeatX from primitives + booleans the same
way; we're just doing a 3-operation toy version to validate the ops compose.

Constructed as:
  outer  := rounded-box SDF  (the casing)
  ch_1   := capsule SDF      (fluid channel 1)
  ch_2   := capsule SDF      (fluid channel 2)
  result := (outer)  -  (ch_1 ∪ ch_2)

Each boolean is one call to picogkgpu.booleans + an auto-reinit pass.

The figure shows mid-plane slices through the SDF after each construction
step, with the zero-isocontour drawn. Times are reported per op so you can
see how the work compounds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from picogkgpu.sparse import SparseSDF  # noqa: E402
from picogkgpu.booleans import union, difference  # noqa: E402


def rounded_box_sdf(n: int, dx: float, half: float, corner_r: float) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    q = np.stack([np.abs(x) - half, np.abs(y) - half, np.abs(z) - half], axis=-1)
    q_pos = np.maximum(q, 0.0)
    inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
    return (np.linalg.norm(q_pos, axis=-1) + inside - corner_r).astype(np.float32)


def capsule_z_sdf(n: int, dx: float, cx: float, cy: float, half_len: float, r: float) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    dx_p = x - cx
    dy_p = y - cy
    dz_p = np.clip(z, -half_len, half_len) - z
    return (np.sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p) - r).astype(np.float32)


def render_steps(slices: list[tuple[str, np.ndarray]], dx: float,
                 out_path: Path, n_steps: int) -> None:
    cols = len(slices)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    vmin, vmax = -10 * dx, 10 * dx
    for ax, (title, phi_slice) in zip(axes, slices):
        im = ax.imshow(phi_slice.T, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.contour(phi_slice.T, levels=[0.0], colors="black", linewidths=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="phi (signed distance)")
    fig.suptitle(
        f"picogk-gpu Phase 1.3 sparse CSG: heat-exchanger-like sketch  "
        f"(mid-plane slices, dx={dx:.4f}, post-op reinit = {n_steps} steps)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    wp.init()
    device = "cuda" if wp.get_device().is_cuda else "cpu"
    print(f"device: {device}")

    n = 128
    dx = 1.0 / (n - 1)
    band = 12 * dx
    print(f"grid: {n}^3 (dx={dx:.5f}, band={band:.4f})")

    # construct primitives as dense, convert to sparse
    t0 = time.perf_counter()
    phi_outer = rounded_box_sdf(n, dx, half=0.30, corner_r=0.03)
    phi_ch1 = capsule_z_sdf(n, dx, cx=-0.10, cy=0.0, half_len=0.25, r=0.05)
    phi_ch2 = capsule_z_sdf(n, dx, cx=+0.10, cy=0.0, half_len=0.25, r=0.05)
    t_build_dense = time.perf_counter() - t0
    print(f"build dense primitives:       {t_build_dense*1000:>7.1f} ms")

    t0 = time.perf_counter()
    s_outer = SparseSDF.from_dense(phi_outer, dx=dx, band_width=band, device=device)
    s_ch1 = SparseSDF.from_dense(phi_ch1, dx=dx, band_width=band, device=device)
    s_ch2 = SparseSDF.from_dense(phi_ch2, dx=dx, band_width=band, device=device)
    t_to_sparse = time.perf_counter() - t0
    print(f"dense -> sparse (3 topologies):{t_to_sparse*1000:>6.1f} ms")
    print(f"  active counts: outer={s_outer.n_active:>7,}  "
          f"ch1={s_ch1.n_active:>7,}  ch2={s_ch2.n_active:>7,}")
    print()

    REINIT_STEPS = 5

    t0 = time.perf_counter()
    s_channels = union(s_ch1, s_ch2, margin=2, auto_reinit=REINIT_STEPS)
    t_union = time.perf_counter() - t0
    print(f"union(ch1, ch2)         + reinit({REINIT_STEPS}): "
          f"{t_union*1000:>7.1f} ms   result active = {s_channels.n_active:,}")

    t0 = time.perf_counter()
    s_result = difference(s_outer, s_channels, margin=2, auto_reinit=REINIT_STEPS)
    t_diff = time.perf_counter() - t0
    print(f"difference(outer, ch) + reinit({REINIT_STEPS}): "
          f"{t_diff*1000:>7.1f} ms   result active = {s_result.n_active:,}")

    print()
    print(f"total CSG wall: {(t_union + t_diff) * 1000:.1f} ms "
          f"({2} booleans, {2*REINIT_STEPS} reinit steps embedded)")

    # render mid-plane slices through each stage
    mid = n // 2
    slices = [
        ("outer (rounded box)", s_outer.to_dense((n, n, n))[:, :, mid]),
        ("union(ch1, ch2)",     s_channels.to_dense((n, n, n))[:, :, mid]),
        ("outer - channels",    s_result.to_dense((n, n, n))[:, :, mid]),
    ]
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"csg_heat_exchanger_sketch_{time.strftime('%Y%m%d_%H%M%S')}.png"
    render_steps(slices, dx, out_path, REINIT_STEPS)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
