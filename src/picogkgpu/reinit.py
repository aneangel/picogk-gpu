"""Hamilton-Jacobi WENO5 level-set reinitialization.

The hot kernel of stock-PicoGK's reinit, identified by perf as ~40% of
HelixHeatX vox=0.5 wall time. Re-distances an SDF so |grad(phi)| approx 1
everywhere while preserving the zero-isocontour.

Two implementations:
  - `reinit_numpy`  : vectorized numpy reference (slow, ground truth)
  - `reinit_warp`   : Warp kernel (GPU, the deliverable)

Both implement the same algorithm (5th-order WENO reconstruction of one-sided
derivatives + Godunov upwind |grad| + Euler time step with smoothed-sign
forcing), so they should agree cell-by-cell within float32 round-off.

Standard refs: Osher & Fedkiw, "Level Set Methods and Dynamic Implicit
Surfaces" (2003), ch. 7; Jiang & Peng, "Weighted ENO Schemes for Hamilton-
Jacobi Equations" (2000).
"""
from __future__ import annotations

import numpy as np
import warp as wp

WENO_EPS = 1e-6


# ── numpy reference ─────────────────────────────────────────────────────────


def _weno5_recon_np(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                    v4: np.ndarray, v5: np.ndarray) -> np.ndarray:
    """WENO5 derivative reconstruction from a 5-point stencil of forward
    differences. Returns the upwind-biased value of df/dx at the center.
    """
    s1 = 13.0 / 12.0 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13.0 / 12.0 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    s3 = 13.0 / 12.0 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2

    a1 = 0.1 / (s1 + WENO_EPS) ** 2
    a2 = 0.6 / (s2 + WENO_EPS) ** 2
    a3 = 0.3 / (s3 + WENO_EPS) ** 2
    asum = a1 + a2 + a3

    q1 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
    q2 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
    q3 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

    return (a1 * q1 + a2 * q2 + a3 * q3) / asum


def _grad_axis_np(phi: np.ndarray, axis: int, dx: float):
    """One-sided WENO5 gradients D- and D+ along `axis` (interior cells only;
    cells closer than 3 to the boundary get nan)."""
    # Forward differences df[k] = (phi[k+1] - phi[k]) / dx along axis.
    fd = (np.roll(phi, -1, axis=axis) - phi) / dx
    # Index helpers along axis
    s = [slice(None)] * 3
    def shift(arr, k):
        s2 = [slice(None)] * 3
        # We compose all stencil neighbours via np.roll for vectorised math.
        return np.roll(arr, -k, axis=axis)

    # Stencil of forward differences around cell i: fd[i-3..i+2].
    f_m3 = shift(fd, -3)
    f_m2 = shift(fd, -2)
    f_m1 = shift(fd, -1)
    f_0  = fd
    f_p1 = shift(fd, +1)
    f_p2 = shift(fd, +2)

    d_minus = _weno5_recon_np(f_m3, f_m2, f_m1, f_0, f_p1)
    d_plus  = _weno5_recon_np(f_p2, f_p1, f_0, f_m1, f_m2)

    return d_minus, d_plus


def reinit_step_numpy(phi: np.ndarray, phi0: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """One Euler step of HJ-WENO5 reinitialization (numpy reference).

    `phi0` is the original signed distance used to compute the smoothed sign
    function; passing it separately keeps the zero-isocontour pinned across
    iterations.
    """
    dm = [None, None, None]
    dp = [None, None, None]
    for ax in range(3):
        dm[ax], dp[ax] = _grad_axis_np(phi, ax, dx)

    sign = phi0 / np.sqrt(phi0 ** 2 + dx ** 2)
    pos = sign > 0.0

    grad_sq = np.zeros_like(phi)
    for ax in range(3):
        a_pos = np.maximum(np.maximum(dm[ax], 0.0) ** 2, np.minimum(dp[ax], 0.0) ** 2)
        a_neg = np.maximum(np.minimum(dm[ax], 0.0) ** 2, np.maximum(dp[ax], 0.0) ** 2)
        grad_sq += np.where(pos, a_pos, a_neg)

    rhs = sign * (np.sqrt(grad_sq) - 1.0)
    out = phi - dt * rhs

    # zero ghost: copy original values into the 3-cell boundary so the test
    # comparison is stable (we don't touch them in the Warp kernel either).
    interior = (slice(3, -3),) * 3
    result = phi.copy()
    result[interior] = out[interior]
    return result


def reinit_numpy(phi0: np.ndarray, dx: float, num_steps: int, dt: float | None = None) -> np.ndarray:
    """Run `num_steps` of HJ-WENO5 reinit on numpy reference. dt defaults to 0.5*dx (CFL)."""
    if dt is None:
        dt = 0.5 * dx
    phi = phi0.astype(np.float32, copy=True)
    for _ in range(num_steps):
        phi = reinit_step_numpy(phi, phi0.astype(np.float32), dx, dt)
    return phi


# ── Warp kernel ─────────────────────────────────────────────────────────────


@wp.func
def _weno5_recon(v1: float, v2: float, v3: float, v4: float, v5: float) -> float:
    eps = float(WENO_EPS)
    s1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2.0 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2.0
    s2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2.0 + 0.25 * (v2 - v4) ** 2.0
    s3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2.0 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2.0

    a1 = 0.1 / (s1 + eps) ** 2.0
    a2 = 0.6 / (s2 + eps) ** 2.0
    a3 = 0.3 / (s3 + eps) ** 2.0
    asum = a1 + a2 + a3

    q1 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
    q2 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
    q3 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

    return (a1 * q1 + a2 * q2 + a3 * q3) / asum


@wp.kernel
def hj_weno5_reinit_step(
    phi_in: wp.array3d(dtype=float),
    phi0: wp.array3d(dtype=float),
    phi_out: wp.array3d(dtype=float),
    dx: float,
    dt: float,
):
    i, j, k = wp.tid()
    nx = phi_in.shape[0]
    ny = phi_in.shape[1]
    nz = phi_in.shape[2]

    if i < 3 or j < 3 or k < 3 or i >= nx - 3 or j >= ny - 3 or k >= nz - 3:
        phi_out[i, j, k] = phi_in[i, j, k]
        return

    inv_dx = 1.0 / dx

    # X-axis one-sided derivatives via WENO5 on forward differences
    fx_m3 = (phi_in[i - 2, j, k] - phi_in[i - 3, j, k]) * inv_dx
    fx_m2 = (phi_in[i - 1, j, k] - phi_in[i - 2, j, k]) * inv_dx
    fx_m1 = (phi_in[i,     j, k] - phi_in[i - 1, j, k]) * inv_dx
    fx_0  = (phi_in[i + 1, j, k] - phi_in[i,     j, k]) * inv_dx
    fx_p1 = (phi_in[i + 2, j, k] - phi_in[i + 1, j, k]) * inv_dx
    fx_p2 = (phi_in[i + 3, j, k] - phi_in[i + 2, j, k]) * inv_dx
    dxm = _weno5_recon(fx_m3, fx_m2, fx_m1, fx_0, fx_p1)
    dxp = _weno5_recon(fx_p2, fx_p1, fx_0, fx_m1, fx_m2)

    fy_m3 = (phi_in[i, j - 2, k] - phi_in[i, j - 3, k]) * inv_dx
    fy_m2 = (phi_in[i, j - 1, k] - phi_in[i, j - 2, k]) * inv_dx
    fy_m1 = (phi_in[i, j,     k] - phi_in[i, j - 1, k]) * inv_dx
    fy_0  = (phi_in[i, j + 1, k] - phi_in[i, j,     k]) * inv_dx
    fy_p1 = (phi_in[i, j + 2, k] - phi_in[i, j + 1, k]) * inv_dx
    fy_p2 = (phi_in[i, j + 3, k] - phi_in[i, j + 2, k]) * inv_dx
    dym = _weno5_recon(fy_m3, fy_m2, fy_m1, fy_0, fy_p1)
    dyp = _weno5_recon(fy_p2, fy_p1, fy_0, fy_m1, fy_m2)

    fz_m3 = (phi_in[i, j, k - 2] - phi_in[i, j, k - 3]) * inv_dx
    fz_m2 = (phi_in[i, j, k - 1] - phi_in[i, j, k - 2]) * inv_dx
    fz_m1 = (phi_in[i, j, k]     - phi_in[i, j, k - 1]) * inv_dx
    fz_0  = (phi_in[i, j, k + 1] - phi_in[i, j, k])     * inv_dx
    fz_p1 = (phi_in[i, j, k + 2] - phi_in[i, j, k + 1]) * inv_dx
    fz_p2 = (phi_in[i, j, k + 3] - phi_in[i, j, k + 2]) * inv_dx
    dzm = _weno5_recon(fz_m3, fz_m2, fz_m1, fz_0, fz_p1)
    dzp = _weno5_recon(fz_p2, fz_p1, fz_0, fz_m1, fz_m2)

    p0 = phi0[i, j, k]
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
    phi_out[i, j, k] = phi_in[i, j, k] - dt * sign * (grad_mag - 1.0)


def reinit_warp(phi0_np: np.ndarray, dx: float, num_steps: int,
                dt: float | None = None, device: str = "cuda") -> np.ndarray:
    """Run `num_steps` of HJ-WENO5 reinit on Warp. Returns numpy array."""
    if dt is None:
        dt = 0.5 * dx
    arr0 = wp.array(phi0_np.astype(np.float32, copy=False), dtype=float, device=device)
    a = wp.array(phi0_np.astype(np.float32, copy=False), dtype=float, device=device)
    b = wp.empty_like(a)

    for _ in range(num_steps):
        wp.launch(
            hj_weno5_reinit_step,
            dim=a.shape,
            inputs=[a, arr0, b, float(dx), float(dt)],
            device=device,
        )
        a, b = b, a

    wp.synchronize()
    return a.numpy()
