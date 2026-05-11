"""HJ-WENO5 level-set reinitialization on a `wp.Volume` (NanoVDB) sparse SDF.

This is the sparse counterpart to `picogkgpu.reinit`. The Volume holds the
narrow-band topology and serves as a coord→index oracle; two flat
`wp.array1d[float]` buffers carry the values and ping-pong each step.

NanoVDB is read-only inside Warp kernels, so we never write into the Volume's
own buffer. Instead, every thread:
  1. Looks up its own (i, j, k) from `coords[tid]` (a parallel array aligned
     to the Volume's internal value ordering).
  2. Reads stencil neighbours via `wp.volume_lookup_index(grid, ...)` →
     `phi_in[idx]`; out-of-band reads return -1, which the kernel replaces
     with a sign-extrapolated band-edge value.
  3. Writes the WENO5+Godunov update to `phi_out[tid]`.

For correctness vs the dense kernel: the two implementations agree cell-by-
cell on the *interior* of the band (3 cells inset from the band edge). At
the band edge they diverge because dense reads real neighbour values while
sparse reads the extrapolated value; this is expected and matches OpenVDB's
own band-edge semantics.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from picogkgpu.reinit import _weno5_recon, WENO_EPS  # noqa: F401 (Warp resolves @wp.func by name)

# bg sentinel passed to load_from_numpy; large enough that
# `lookup_f(inactive) >> band_width` so we can detect it in the kernel.
DEFAULT_BG = 1.0e10
BG_DETECT = 1.0e6  # any returned value >= this is treated as the bg sentinel


@wp.func
def _vload(grid: wp.uint64, ii: int, jj: int, kk: int,
           phi_in: wp.array1d(dtype=float), band_extrap: float) -> float:
    """Read phi at integer coord (ii,jj,kk); fall back to band_extrap if inactive."""
    idx = wp.volume_lookup_index(grid, ii, jj, kk)
    if idx < 0:
        return band_extrap
    return phi_in[idx]


@wp.kernel
def hj_weno5_reinit_step_sparse(
    grid: wp.uint64,
    coords: wp.array2d(dtype=int),       # (N, 3) — Volume's active coords
    phi_in: wp.array1d(dtype=float),     # (N,)   — values, aligned to coords
    phi0:   wp.array1d(dtype=float),     # (N,)   — initial values (for sign)
    phi_out: wp.array1d(dtype=float),    # (N,)   — output
    dx: float,
    dt: float,
    band: float,
    bg_detect: float,
):
    tid = wp.tid()
    i = coords[tid, 0]
    j = coords[tid, 1]
    k = coords[tid, 2]

    inv_dx = 1.0 / dx
    p0 = phi0[tid]
    sign_p0 = wp.sign(p0)
    band_extrap = sign_p0 * band

    px_m3 = _vload(grid, i - 3, j, k, phi_in, band_extrap)
    px_m2 = _vload(grid, i - 2, j, k, phi_in, band_extrap)
    px_m1 = _vload(grid, i - 1, j, k, phi_in, band_extrap)
    px_0  = phi_in[tid]
    px_p1 = _vload(grid, i + 1, j, k, phi_in, band_extrap)
    px_p2 = _vload(grid, i + 2, j, k, phi_in, band_extrap)
    px_p3 = _vload(grid, i + 3, j, k, phi_in, band_extrap)

    fx_m3 = (px_m2 - px_m3) * inv_dx
    fx_m2 = (px_m1 - px_m2) * inv_dx
    fx_m1 = (px_0  - px_m1) * inv_dx
    fx_0  = (px_p1 - px_0)  * inv_dx
    fx_p1 = (px_p2 - px_p1) * inv_dx
    fx_p2 = (px_p3 - px_p2) * inv_dx
    dxm = _weno5_recon(fx_m3, fx_m2, fx_m1, fx_0, fx_p1)
    dxp = _weno5_recon(fx_p2, fx_p1, fx_0, fx_m1, fx_m2)

    py_m3 = _vload(grid, i, j - 3, k, phi_in, band_extrap)
    py_m2 = _vload(grid, i, j - 2, k, phi_in, band_extrap)
    py_m1 = _vload(grid, i, j - 1, k, phi_in, band_extrap)
    py_0  = phi_in[tid]
    py_p1 = _vload(grid, i, j + 1, k, phi_in, band_extrap)
    py_p2 = _vload(grid, i, j + 2, k, phi_in, band_extrap)
    py_p3 = _vload(grid, i, j + 3, k, phi_in, band_extrap)
    fy_m3 = (py_m2 - py_m3) * inv_dx
    fy_m2 = (py_m1 - py_m2) * inv_dx
    fy_m1 = (py_0  - py_m1) * inv_dx
    fy_0  = (py_p1 - py_0)  * inv_dx
    fy_p1 = (py_p2 - py_p1) * inv_dx
    fy_p2 = (py_p3 - py_p2) * inv_dx
    dym = _weno5_recon(fy_m3, fy_m2, fy_m1, fy_0, fy_p1)
    dyp = _weno5_recon(fy_p2, fy_p1, fy_0, fy_m1, fy_m2)

    pz_m3 = _vload(grid, i, j, k - 3, phi_in, band_extrap)
    pz_m2 = _vload(grid, i, j, k - 2, phi_in, band_extrap)
    pz_m1 = _vload(grid, i, j, k - 1, phi_in, band_extrap)
    pz_0  = phi_in[tid]
    pz_p1 = _vload(grid, i, j, k + 1, phi_in, band_extrap)
    pz_p2 = _vload(grid, i, j, k + 2, phi_in, band_extrap)
    pz_p3 = _vload(grid, i, j, k + 3, phi_in, band_extrap)
    fz_m3 = (pz_m2 - pz_m3) * inv_dx
    fz_m2 = (pz_m1 - pz_m2) * inv_dx
    fz_m1 = (pz_0  - pz_m1) * inv_dx
    fz_0  = (pz_p1 - pz_0)  * inv_dx
    fz_p1 = (pz_p2 - pz_p1) * inv_dx
    fz_p2 = (pz_p3 - pz_p2) * inv_dx
    dzm = _weno5_recon(fz_m3, fz_m2, fz_m1, fz_0, fz_p1)
    dzp = _weno5_recon(fz_p2, fz_p1, fz_0, fz_m1, fz_m2)

    sign = p0 / wp.sqrt(p0 * p0 + dx * dx)

    if sign > 0.0:
        ax = wp.max(wp.max(dxm, 0.0) ** 2.0, wp.min(dxp, 0.0) ** 2.0)
        ay = wp.max(wp.max(dym, 0.0) ** 2.0, wp.min(dyp, 0.0) ** 2.0)
        az = wp.max(wp.max(dzm, 0.0) ** 2.0, wp.min(dzp, 0.0) ** 2.0)
    else:
        ax = wp.max(wp.min(dxm, 0.0) ** 2.0, wp.max(dxp, 0.0) ** 2.0)
        ay = wp.max(wp.min(dym, 0.0) ** 2.0, wp.max(dyp, 0.0) ** 2.0)
        az = wp.max(wp.min(dzm, 0.0) ** 2.0, wp.max(dzp, 0.0) ** 2.0)

    grad_mag = wp.sqrt(ax + ay + az)
    phi_out[tid] = phi_in[tid] - dt * sign * (grad_mag - 1.0)


class SparseReinit:
    """Reusable sparse HJ-WENO5 reinit on a NanoVDB-backed wp.Volume.

    `phi_dense` is the input SDF; only cells with `|phi| < band_width` are
    activated in the Volume topology. The Volume is fixed for the life of
    this object; iterate `run(num_steps)` to evolve the values in place.
    """

    def __init__(self, phi_dense: np.ndarray, dx: float, band_width: float,
                 dt: float | None = None, device: str = "cuda") -> None:
        if phi_dense.ndim != 3:
            raise ValueError("phi_dense must be a 3D array")
        self.dx = float(dx)
        self.dt = float(dt) if dt is not None else 0.5 * self.dx
        self.band_width = float(band_width)
        self.device = device

        # True sparsity: only allocate voxels actually in the narrow band.
        # `Volume.load_from_numpy` is dense regardless of bg_value, so use
        # `allocate_by_voxels` with an explicit coord list.
        mask = np.abs(phi_dense) < band_width
        if not mask.any():
            raise ValueError("no active cells with |phi| < band_width")
        coords_in = np.argwhere(mask).astype(np.int32)
        coords_in_wp = wp.array(coords_in, dtype=wp.vec3i, device=device)
        self.volume = wp.Volume.allocate_by_voxels(
            voxel_points=coords_in_wp, voxel_size=self.dx, device=device,
        )
        # NanoVDB returns coords in its canonical sorted ordering aligned with
        # the active value buffer. Use that as the source of truth.
        self.coords = self.volume.get_voxels()
        self.n_active = int(self.coords.shape[0])

        coords_np = self.coords.numpy()
        vals = phi_dense[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]].astype(np.float32)
        # defensive clip in case any returned coord wandered outside the band
        out_of_band = np.abs(vals) >= band_width
        vals = np.where(out_of_band, np.sign(vals).astype(np.float32) * band_width, vals)
        vals = vals.astype(np.float32)

        self.phi_a = wp.array(vals, dtype=float, device=device)
        self.phi_b = wp.empty_like(self.phi_a)
        self.phi0 = wp.array(vals, dtype=float, device=device)
        self._graphs: dict[int, object | None] = {}

        # warm-up JIT
        wp.launch(
            hj_weno5_reinit_step_sparse, dim=self.n_active,
            inputs=[self.volume.id, self.coords, self.phi_a, self.phi0, self.phi_b,
                    self.dx, self.dt, self.band_width, BG_DETECT],
            device=device,
        )
        wp.synchronize()

    def _build_graph(self, num_steps: int) -> None:
        if not wp.get_device(self.device).is_cuda:
            self._graphs[num_steps] = None
            return
        with wp.ScopedCapture(device=self.device) as cap:
            a, b = self.phi_a, self.phi_b
            for _ in range(num_steps):
                wp.launch(
                    hj_weno5_reinit_step_sparse, dim=self.n_active,
                    inputs=[self.volume.id, self.coords, a, self.phi0, b,
                            self.dx, self.dt, self.band_width, BG_DETECT],
                    device=self.device,
                )
                a, b = b, a
        self._graphs[num_steps] = cap.graph

    def run(self, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Run `num_steps` reinit iterations. Returns (coords_np, values_np)."""
        if num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        # reset values from phi0 each call so the op is functionally pure
        wp.copy(self.phi_a, self.phi0)

        if num_steps == 0:
            wp.synchronize()
            return self.coords.numpy(), self.phi_a.numpy()

        if num_steps not in self._graphs:
            self._build_graph(num_steps)
        graph = self._graphs[num_steps]
        if graph is not None:
            wp.capture_launch(graph)
        else:
            a, b = self.phi_a, self.phi_b
            for _ in range(num_steps):
                wp.launch(
                    hj_weno5_reinit_step_sparse, dim=self.n_active,
                    inputs=[self.volume.id, self.coords, a, self.phi0, b,
                            self.dx, self.dt, self.band_width, BG_DETECT],
                    device=self.device,
                )
                a, b = b, a
            self.phi_a, self.phi_b = a, b

        wp.synchronize()
        out_buf = self.phi_a if num_steps % 2 == 0 else self.phi_b
        return self.coords.numpy(), out_buf.numpy()
