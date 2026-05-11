"""Sparse SDF boolean operations: union, intersection, difference.

The standard SDF formulae:
  - union(A, B)        = min(phi_A, phi_B)
  - intersect(A, B)    = max(phi_A, phi_B)
  - difference(A, B)   = max(phi_A, -phi_B)

Topology growth strategy (Phase 1.3, "pre-allocate + margin"):

  Union may produce a result whose narrow band extends past either input's
  band. We allocate the result topology as the Chebyshev-distance dilation
  of (A's coords ∪ B's coords) by `margin` voxels. This is conservative
  (some over-allocation in regions where the result band isn't actually
  needed) but correct for any A,B whose interiors are fully covered by
  their narrow bands.

  Intersection and difference don't grow the band — but we still use the
  same dilated-union topology to keep the kernel signatures uniform and to
  guarantee enough margin for a subsequent reinit pass.

Cells in the result topology that lie outside one input's band fall back to
+band_width for that input (i.e. we assume "outside the solid"). This is
**only correct when every input's interior is fully covered by its narrow
band** — i.e. `band_width >= max(|phi|)` over the input cells. Pass a
band wide enough to contain the deepest interior, or Phase 1.4 will add
sign-propagation so deep interiors can sit outside the band correctly.

After the boolean min/max, the field is not a true SDF away from the new
zero-isocontour, so we reinitialize before returning (matching OpenVDB's
behavior). Pass `auto_reinit=0` to skip and inspect raw output.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from picogkgpu.sparse import SparseSDF


OP_UNION = wp.constant(0)
OP_INTERSECT = wp.constant(1)
OP_DIFFERENCE = wp.constant(2)


@wp.kernel
def _boolean_combine_kernel(
    grid_a: wp.uint64,
    grid_b: wp.uint64,
    coords_r: wp.array2d(dtype=int),
    phi_a: wp.array1d(dtype=float),
    phi_b: wp.array1d(dtype=float),
    phi_r: wp.array1d(dtype=float),
    op: int,
    band: float,
):
    tid = wp.tid()
    i = coords_r[tid, 0]
    j = coords_r[tid, 1]
    k = coords_r[tid, 2]

    idx_a = wp.volume_lookup_index(grid_a, i, j, k)
    if idx_a >= 0:
        va = phi_a[idx_a]
    else:
        va = band

    idx_b = wp.volume_lookup_index(grid_b, i, j, k)
    if idx_b >= 0:
        vb = phi_b[idx_b]
    else:
        vb = band

    if op == 0:
        result = wp.min(va, vb)
    elif op == 1:
        result = wp.max(va, vb)
    else:
        result = wp.max(va, -vb)

    phi_r[tid] = wp.clamp(result, -band, band)


_COORD_BITS = 20                 # supports grids up to 2^20 = 1,048,576 per axis
_COORD_MASK = (1 << _COORD_BITS) - 1


def _pack_coords(coords: np.ndarray) -> np.ndarray:
    """Pack (N, 3) int32 coords into (N,) int64 for fast np.unique on 1D."""
    c = coords.astype(np.int64)
    return ((c[:, 0] & _COORD_MASK) << (2 * _COORD_BITS)) | \
           ((c[:, 1] & _COORD_MASK) << _COORD_BITS) | \
           (c[:, 2] & _COORD_MASK)


def _unpack_coords(packed: np.ndarray) -> np.ndarray:
    i = ((packed >> (2 * _COORD_BITS)) & _COORD_MASK).astype(np.int32)
    j = ((packed >> _COORD_BITS) & _COORD_MASK).astype(np.int32)
    k = (packed & _COORD_MASK).astype(np.int32)
    return np.stack([i, j, k], axis=1)


def _unique_coords(coords: np.ndarray) -> np.ndarray:
    """Equivalent to np.unique(coords, axis=0) but ~10–50x faster via int64 packing."""
    return _unpack_coords(np.unique(_pack_coords(coords)))


def _dilate_coords_chebyshev(coords: np.ndarray, margin: int) -> np.ndarray:
    """Set of integer coords within Chebyshev distance `margin` of any input."""
    if margin == 0:
        return _unique_coords(coords)
    rng = np.arange(-margin, margin + 1, dtype=np.int32)
    di, dj, dk = np.meshgrid(rng, rng, rng, indexing="ij")
    offs = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)  # ((2m+1)^3, 3)
    expanded = coords[:, None, :] + offs[None, :, :]
    expanded = expanded.reshape(-1, 3)
    return _unique_coords(expanded)


def _result_topology(a: SparseSDF, b: SparseSDF, op: int, margin: int) -> np.ndarray:
    """Coord list for the result. Each op has the minimal correct topology:
      - union     : A.coords ∪ B.coords  (result band is union of input bands)
      - intersect : A.coords ∪ B.coords  (result is subset, kernel clamps)
      - difference: A.coords             (result is contained in A's band)
    """
    a_coords = a.coords.numpy()
    if op == 2:  # difference: only A's topology needed
        coords = a_coords
    else:        # union, intersect: both
        coords = np.concatenate([a_coords, b.coords.numpy()], axis=0)
    if margin > 0:
        coords = _dilate_coords_chebyshev(coords, margin)
    else:
        coords = _unique_coords(coords)
    return coords.astype(np.int32)


def _check_compatible(a: SparseSDF, b: SparseSDF) -> None:
    if abs(a.dx - b.dx) > 1e-9:
        raise ValueError(f"SDF dx mismatch: {a.dx} vs {b.dx}")
    if a.device != b.device:
        raise ValueError(f"SDF device mismatch: {a.device} vs {b.device}")


def _apply_boolean(a: SparseSDF, b: SparseSDF, op: int,
                   margin: int, auto_reinit: int) -> SparseSDF:
    _check_compatible(a, b)
    band = max(a.band_width, b.band_width)

    coords_r = _result_topology(a, b, op, margin)
    # Allocate the result volume; fill values via the combine kernel.
    coords_r_wp = wp.array(coords_r, dtype=wp.vec3i, device=a.device)
    volume_r = wp.Volume.allocate_by_voxels(
        voxel_points=coords_r_wp, voxel_size=a.dx, device=a.device,
    )
    coords_canonical = volume_r.get_voxels()
    n_active = int(coords_canonical.shape[0])
    values_r = wp.empty(n_active, dtype=float, device=a.device)

    wp.launch(
        _boolean_combine_kernel,
        dim=n_active,
        inputs=[
            a.volume.id, b.volume.id, coords_canonical,
            a.values, b.values, values_r,
            op, band,
        ],
        device=a.device,
    )
    wp.synchronize()
    result = SparseSDF(volume_r, coords_canonical, values_r, a.dx, band, a.device)
    if auto_reinit > 0:
        result = result.reinit(num_steps=auto_reinit)
    return result


def union(a: SparseSDF, b: SparseSDF, margin: int = 1,
          auto_reinit: int = 3) -> SparseSDF:
    """A ∪ B. Default margin=1 voxel for safety near band edges."""
    return _apply_boolean(a, b, op=0, margin=margin, auto_reinit=auto_reinit)


def intersection(a: SparseSDF, b: SparseSDF, margin: int = 1,
                 auto_reinit: int = 3) -> SparseSDF:
    """A ∩ B."""
    return _apply_boolean(a, b, op=1, margin=margin, auto_reinit=auto_reinit)


def difference(a: SparseSDF, b: SparseSDF, margin: int = 0,
               auto_reinit: int = 3) -> SparseSDF:
    """A − B. Default margin=0: difference is contained in A's topology."""
    return _apply_boolean(a, b, op=2, margin=margin, auto_reinit=auto_reinit)
