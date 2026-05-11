"""Validate sparse boolean ops against a numpy ground-truth reference.

For each op we build two simple SDFs (sphere-shaped, easy to reason about),
run the GPU boolean, and compare its zero-isocontour to the analytic
expected shape on a dense grid sampled at the same dx.

We compare on the *interior* of the result band (margin from the band edge)
because the sign-extrapolation at the boundary is an approximation.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from picogkgpu.sparse import SparseSDF
from picogkgpu.booleans import union, intersection, difference


def _sphere_sdf(n: int, dx: float, cx: float, cy: float, cz: float, radius: float) -> np.ndarray:
    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return (np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius).astype(np.float32)


@pytest.fixture(scope="module")
def warp_device():
    wp.init()
    return "cuda" if wp.get_device().is_cuda else "cpu"


def _interior_compare(phi_got_dense: np.ndarray, phi_ref_dense: np.ndarray,
                      band: float, interior_margin: float, atol: float) -> None:
    """Compare values only where the reference SDF says we're well inside the band."""
    mask = np.abs(phi_ref_dense) < (band - interior_margin)
    assert mask.sum() > 50, "test would compare too few cells"
    np.testing.assert_allclose(phi_got_dense[mask], phi_ref_dense[mask], atol=atol, rtol=atol)


def test_union_two_overlapping_spheres(warp_device):
    # band must be wide enough to fully contain each input's interior;
    # see picogkgpu.booleans docstring on the sign-propagation limit.
    n, dx = 96, 1.0 / 48.0
    band = 12 * dx  # 0.25 — exceeds the deepest interior used below
    # two spheres of radius 0.15 at +/- 0.08 along x, so they overlap;
    # 0.15 < band (0.25) so the entire interior fits in the band
    phi_a = _sphere_sdf(n, dx, -0.08, 0, 0, 0.15)
    phi_b = _sphere_sdf(n, dx, +0.08, 0, 0, 0.15)
    # analytic union sdf is min(phi_a, phi_b); not a true SDF away from the
    # zero contour, but its zero contour is exactly the union surface
    ref_union = np.minimum(phi_a, phi_b)

    a = SparseSDF.from_dense(phi_a, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_b, dx=dx, band_width=band, device=warp_device)
    got = union(a, b, margin=2, auto_reinit=0).to_dense((n, n, n))

    _interior_compare(got, ref_union, band=band, interior_margin=2 * dx, atol=2 * dx)


def test_intersection_two_overlapping_spheres(warp_device):
    # band must be wide enough to fully contain each input's interior;
    # see picogkgpu.booleans docstring on the sign-propagation limit.
    n, dx = 96, 1.0 / 48.0
    band = 12 * dx  # 0.25 — exceeds the deepest interior used below
    phi_a = _sphere_sdf(n, dx, -0.05, 0, 0, 0.15)
    phi_b = _sphere_sdf(n, dx, +0.05, 0, 0, 0.15)
    ref_inter = np.maximum(phi_a, phi_b)

    a = SparseSDF.from_dense(phi_a, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_b, dx=dx, band_width=band, device=warp_device)
    got = intersection(a, b, margin=2, auto_reinit=0).to_dense((n, n, n))

    _interior_compare(got, ref_inter, band=band, interior_margin=2 * dx, atol=2 * dx)


def test_difference_carves_sphere_out_of_sphere(warp_device):
    # band must be wide enough to fully contain each input's interior;
    # see picogkgpu.booleans docstring on the sign-propagation limit.
    n, dx = 96, 1.0 / 48.0
    band = 12 * dx  # 0.25 — exceeds the deepest interior used below
    phi_a = _sphere_sdf(n, dx, 0, 0, 0, 0.18)        # big sphere, fits in band 0.25
    phi_b = _sphere_sdf(n, dx, 0.08, 0, 0, 0.10)     # small sphere off-center
    ref_diff = np.maximum(phi_a, -phi_b)             # A - B

    a = SparseSDF.from_dense(phi_a, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_b, dx=dx, band_width=band, device=warp_device)
    got = difference(a, b, margin=2, auto_reinit=0).to_dense((n, n, n))

    _interior_compare(got, ref_diff, band=band, interior_margin=2 * dx, atol=2 * dx)


def test_union_reinit_preserves_zero_contour(warp_device):
    """Union followed by reinit should preserve the zero-isocontour location."""
    # band must be wide enough to fully contain each input's interior;
    # see picogkgpu.booleans docstring on the sign-propagation limit.
    n, dx = 96, 1.0 / 48.0
    band = 12 * dx  # 0.25 — exceeds the deepest interior used below
    phi_a = _sphere_sdf(n, dx, -0.07, 0, 0, 0.13)
    phi_b = _sphere_sdf(n, dx, +0.07, 0, 0, 0.13)
    ref_union = np.minimum(phi_a, phi_b)

    a = SparseSDF.from_dense(phi_a, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_b, dx=dx, band_width=band, device=warp_device)
    # do auto_reinit=5 — should re-distance without moving the zero contour
    got = union(a, b, margin=2, auto_reinit=5).to_dense((n, n, n))

    # check sign agreement: where ref is clearly inside (<-2*dx) or outside (>2*dx),
    # the GPU result should have the same sign
    clearly_inside = ref_union < -2 * dx
    clearly_outside = ref_union > 2 * dx
    assert np.all(got[clearly_inside] < 0)
    assert np.all(got[clearly_outside] > 0)


def test_thick_interior_difference(warp_device):
    """A thick cube minus a small sphere — the cube interior is far deeper
    than the band, so the boolean kernel needs Phase 1.4's bg_sign field to
    classify cells outside the band as "inside cube" vs "outside cube".

    Without sign propagation, cells deep inside the cube interior (outside
    the narrow band) get treated as `+band` (outside solid), which makes
    `difference` return the wrong sign at the sphere/cube interface.
    """
    n, dx = 64, 1.0 / 32.0
    band = 4 * dx  # ~0.125, much smaller than cube half-extent (0.30)

    coords = (np.arange(n) - (n - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    # cube of half-extent 0.30: interior cells have phi down to -0.30
    phi_cube = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z)).astype(np.float32) - 0.30
    # small sphere off-center, fully contained in cube and band
    phi_sph = (np.sqrt((x - 0.10) ** 2 + y ** 2 + z ** 2) - 0.07).astype(np.float32)

    ref_diff = np.maximum(phi_cube, -phi_sph)

    a = SparseSDF.from_dense(phi_cube, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_sph, dx=dx, band_width=band, device=warp_device)
    assert a.bg_sign is not None, "Phase 1.4 should populate bg_sign on from_dense"

    got = difference(a, b, margin=1, auto_reinit=0).to_dense((n, n, n))

    # Compare at cells that are well-inside the result's band (i.e., near the
    # cube boundary and near the sphere boundary). These are the cells where
    # the sign-propagation fix matters most.
    mask = np.abs(ref_diff) < (band - 2 * dx)
    assert mask.sum() > 100
    np.testing.assert_allclose(got[mask], ref_diff[mask], atol=2 * dx, rtol=2 * dx)


def test_topology_growth_via_margin(warp_device):
    """The result topology should grow by approximately `margin` voxels in each
    direction beyond the union of inputs."""
    n, dx = 48, 1.0 / 24.0
    band = 6 * dx
    phi_a = _sphere_sdf(n, dx, 0, 0, 0, 0.18)
    phi_b = _sphere_sdf(n, dx, 0.05, 0, 0, 0.10)

    a = SparseSDF.from_dense(phi_a, dx=dx, band_width=band, device=warp_device)
    b = SparseSDF.from_dense(phi_b, dx=dx, band_width=band, device=warp_device)
    margin = 2
    got = union(a, b, margin=margin, auto_reinit=0)

    # the result topology should have at least as many active cells as the union
    # of the inputs, and no more than (1 + 2*margin)^3 times more (loose bound)
    n_union_min = max(a.n_active, b.n_active)
    n_loose_max = (a.n_active + b.n_active) * (2 * margin + 1) ** 3
    assert got.n_active >= n_union_min, f"result smaller than larger input ({got.n_active} < {n_union_min})"
    assert got.n_active <= n_loose_max
