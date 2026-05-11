"""Validate Warp HJ-WENO5 reinit against the numpy reference.

Strategy: build a perturbed sphere SDF (|grad| neq 1 by construction), run
N steps of reinit through both implementations, assert agreement cell-by-cell
within numerical tolerance. Grid is small (48^3) so the numpy reference runs
in seconds.

Also includes a convergence sanity check: for a true sphere SDF (already
|grad| = 1), one reinit step should leave the field essentially unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from picogkgpu.reinit import reinit_numpy, reinit_warp


def _sphere_sdf(n: int, dx: float, radius: float, perturb_scale: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """Centered-sphere SDF on an [n,n,n] grid with optional radial perturbation
    that keeps the zero-isocontour but warps |grad| away from 1."""
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    phi = r - radius
    if perturb_scale != 0.0:
        rng = np.random.default_rng(seed)
        # multiply by a smooth field bounded away from 0 so we don't move the zero-contour
        noise = 1.0 + perturb_scale * rng.standard_normal(phi.shape).astype(np.float32)
        phi = phi * noise.astype(np.float32)
    return phi.astype(np.float32)


@pytest.fixture(scope="module")
def warp_device():
    wp.init()
    dev = "cuda" if wp.get_device().is_cuda else "cpu"
    return dev


def test_warp_matches_numpy_one_step(warp_device):
    n, dx = 48, 1.0 / 24.0
    phi0 = _sphere_sdf(n, dx, radius=0.4, perturb_scale=0.15, seed=42)
    out_np = reinit_numpy(phi0, dx=dx, num_steps=1)
    out_wp = reinit_warp(phi0, dx=dx, num_steps=1, device=warp_device)

    # Interior cells only; both implementations leave the 3-cell boundary as input.
    interior = (slice(3, -3),) * 3
    np.testing.assert_allclose(out_wp[interior], out_np[interior], atol=1e-4, rtol=1e-4)


def test_warp_matches_numpy_multi_step(warp_device):
    n, dx = 48, 1.0 / 24.0
    phi0 = _sphere_sdf(n, dx, radius=0.4, perturb_scale=0.15, seed=42)
    out_np = reinit_numpy(phi0, dx=dx, num_steps=5)
    out_wp = reinit_warp(phi0, dx=dx, num_steps=5, device=warp_device)

    interior = (slice(3, -3),) * 3
    # accumulated round-off over 5 steps loosens tolerance slightly
    np.testing.assert_allclose(out_wp[interior], out_np[interior], atol=2e-3, rtol=2e-3)


def test_clean_sphere_is_fixed_point(warp_device):
    """A perfect sphere SDF has |grad|=1 already; reinit should be near-identity."""
    n, dx = 48, 1.0 / 24.0
    phi0 = _sphere_sdf(n, dx, radius=0.4, perturb_scale=0.0)
    out = reinit_warp(phi0, dx=dx, num_steps=3, device=warp_device)

    interior = (slice(3, -3),) * 3
    delta = np.abs(out[interior] - phi0[interior]).max()
    assert delta < 5e-3, f"clean sphere drifted by {delta} after 3 reinit steps"
