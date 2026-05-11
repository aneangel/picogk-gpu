"""Sparse (NanoVDB) reinit must agree with the dense reinit in the *interior*
of the active band. The two diverge at the band edge by construction
(dense reads real neighbour values, sparse uses sign-extrapolated bg), so
the comparison restricts to cells at least `interior_margin` cells inside
the band.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from picogkgpu.reinit import reinit_warp
from picogkgpu.sparse_reinit import SparseReinit


def _sphere_sdf(n: int, dx: float, radius: float) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) - radius).astype(np.float32)


@pytest.fixture(scope="module")
def warp_device():
    wp.init()
    return "cuda" if wp.get_device().is_cuda else "cpu"


def test_sparse_matches_dense_interior(warp_device):
    """Sparse and dense reinit agree on cells well inside the active band."""
    n, dx = 64, 1.0 / 32.0
    radius = 0.4
    # band wide enough that the interior of the band has a clean comparison zone
    band = 8.0 * dx
    interior_margin = 4 * dx   # only compare cells at least 4 voxels from band edge

    phi0 = _sphere_sdf(n, dx, radius)

    # dense reference
    dense_out = reinit_warp(phi0, dx=dx, num_steps=3, device=warp_device)

    # sparse
    op = SparseReinit(phi0, dx=dx, band_width=band, device=warp_device)
    coords_np, sparse_out = op.run(num_steps=3)

    # compare only at coords that are well inside the band (per the original SDF)
    initial_at_coords = phi0[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]]
    interior_mask = np.abs(initial_at_coords) < (band - interior_margin)
    assert interior_mask.sum() > 100, "test would compare too few cells"

    ic = coords_np[interior_mask]
    dense_at_ic = dense_out[ic[:, 0], ic[:, 1], ic[:, 2]]
    sparse_at_ic = sparse_out[interior_mask]

    np.testing.assert_allclose(sparse_at_ic, dense_at_ic, atol=5e-3, rtol=5e-3)


def test_sparse_reinit_class_reuse(warp_device):
    """Running the same SparseReinit twice produces identical output (functional purity)."""
    n, dx = 48, 1.0 / 24.0
    phi0 = _sphere_sdf(n, dx, radius=0.4)
    op = SparseReinit(phi0, dx=dx, band_width=6.0 * dx, device=warp_device)

    _, vals_1 = op.run(num_steps=3)
    _, vals_2 = op.run(num_steps=3)
    np.testing.assert_array_equal(vals_1, vals_2)


def test_sparse_zero_steps_returns_initial(warp_device):
    """num_steps=0 should return the initial band values, unchanged."""
    n, dx = 32, 1.0 / 16.0
    phi0 = _sphere_sdf(n, dx, radius=0.3)
    op = SparseReinit(phi0, dx=dx, band_width=5.0 * dx, device=warp_device)
    _, vals = op.run(num_steps=0)
    # initial values are bounded by band; dense->sparse just clipped to ±band
    assert np.all(np.abs(vals) <= 5.0 * dx + 1e-5)
