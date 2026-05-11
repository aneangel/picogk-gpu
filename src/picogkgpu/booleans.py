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


@wp.func
def _read_with_sign(
    grid: wp.uint64,
    bg_sign: wp.array3d(dtype=wp.int32),
    bg_origin_i: int, bg_origin_j: int, bg_origin_k: int,
    bg_leaf_size: int,
    bg_dim_i: int, bg_dim_j: int, bg_dim_k: int,
    phi: wp.array1d(dtype=float),
    i: int, j: int, k: int,
    band: float,
) -> float:
    """Read phi at coord. If in band: return value. If out-of-band: consult
    background-sign field for the correct ±band fallback."""
    idx = wp.volume_lookup_index(grid, i, j, k)
    if idx >= 0:
        return phi[idx]
    # outside the band — look up coarse leaf sign
    li = (i - bg_origin_i) // bg_leaf_size
    lj = (j - bg_origin_j) // bg_leaf_size
    lk = (k - bg_origin_k) // bg_leaf_size
    if li < 0 or lj < 0 or lk < 0 or li >= bg_dim_i or lj >= bg_dim_j or lk >= bg_dim_k:
        return band  # outside the bg field's coverage — assume "outside solid"
    s = bg_sign[li, lj, lk]
    if s < 0:
        return -band
    return band


@wp.kernel
def _boolean_combine_kernel(
    grid_a: wp.uint64,
    grid_b: wp.uint64,
    bg_sign_a: wp.array3d(dtype=wp.int32),
    bg_origin_a_i: int, bg_origin_a_j: int, bg_origin_a_k: int,
    bg_dim_a_i: int, bg_dim_a_j: int, bg_dim_a_k: int,
    bg_sign_b: wp.array3d(dtype=wp.int32),
    bg_origin_b_i: int, bg_origin_b_j: int, bg_origin_b_k: int,
    bg_dim_b_i: int, bg_dim_b_j: int, bg_dim_b_k: int,
    bg_leaf_size: int,
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

    va = _read_with_sign(grid_a, bg_sign_a,
                          bg_origin_a_i, bg_origin_a_j, bg_origin_a_k, bg_leaf_size,
                          bg_dim_a_i, bg_dim_a_j, bg_dim_a_k,
                          phi_a, i, j, k, band)
    vb = _read_with_sign(grid_b, bg_sign_b,
                          bg_origin_b_i, bg_origin_b_j, bg_origin_b_k, bg_leaf_size,
                          bg_dim_b_i, bg_dim_b_j, bg_dim_b_k,
                          phi_b, i, j, k, band)

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
    """Coord list for the result.

    All three ops use A.coords ∪ B.coords as the base topology. Even
    `difference` needs B's coords: cells in A's deep interior that lie near
    B's surface will have |max(phi_a, -phi_b)| < band (driven by B's value)
    and must be in the result topology, even though they're outside A's
    own band.
    """
    a_coords = a.coords.numpy()
    b_coords = b.coords.numpy()
    coords = np.concatenate([a_coords, b_coords], axis=0)
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


def _combine_bg_sign(a: SparseSDF, b: SparseSDF, op: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Compute the result's background-sign field by combining inputs' signs.

    Aligns both inputs onto a common origin (the elementwise-min of their
    origins) and a common shape (covering both). Cells outside an input's
    coverage are treated as +1 (exterior).
    """
    sa = a.bg_sign.numpy() if a.bg_sign is not None else None
    sb = b.bg_sign.numpy() if b.bg_sign is not None else None
    if sa is None and sb is None:
        return None, (0, 0, 0)
    leaf = a.bg_leaf_size

    # axis-aligned bounding box of both sign fields, in leaf-coord space
    a_lo = np.array(a.bg_origin, dtype=np.int64) // leaf if sa is not None else np.array([0, 0, 0], dtype=np.int64)
    b_lo = np.array(b.bg_origin, dtype=np.int64) // leaf if sb is not None else np.array([0, 0, 0], dtype=np.int64)
    a_hi = a_lo + np.array(sa.shape, dtype=np.int64) if sa is not None else a_lo
    b_hi = b_lo + np.array(sb.shape, dtype=np.int64) if sb is not None else b_lo
    lo = np.minimum(a_lo, b_lo)
    hi = np.maximum(a_hi, b_hi)
    shape = tuple(int(v) for v in (hi - lo))
    if any(s <= 0 for s in shape):
        return None, (0, 0, 0)

    # Default to "outside" (+1) in unioned coverage
    SA = np.full(shape, 1, dtype=np.int8)
    SB = np.full(shape, 1, dtype=np.int8)
    if sa is not None:
        off = a_lo - lo
        SA[off[0]:off[0] + sa.shape[0], off[1]:off[1] + sa.shape[1], off[2]:off[2] + sa.shape[2]] = sa
    if sb is not None:
        off = b_lo - lo
        SB[off[0]:off[0] + sb.shape[0], off[1]:off[1] + sb.shape[1], off[2]:off[2] + sb.shape[2]] = sb

    if op == 0:  # union: inside if either inside; outside if both outside
        result = np.where(
            (SA == -1) | (SB == -1), -1,
            np.where((SA == 1) & (SB == 1), 1, 0),
        ).astype(np.int8)
    elif op == 1:  # intersection: inside only if both inside
        result = np.where(
            (SA == -1) & (SB == -1), -1,
            np.where((SA == 1) | (SB == 1), 1, 0),
        ).astype(np.int8)
    else:  # difference (A - B): inside iff A inside AND B outside
        result = np.where(
            (SA == -1) & (SB == 1), -1,
            np.where((SA == 1) | (SB == -1), 1, 0),
        ).astype(np.int8)

    origin = tuple(int(v) for v in (lo * leaf))
    return result, origin


def _apply_boolean(a: SparseSDF, b: SparseSDF, op: int,
                   margin: int, auto_reinit: int) -> SparseSDF:
    _check_compatible(a, b)
    band = max(a.band_width, b.band_width)

    coords_r = _result_topology(a, b, op, margin)
    coords_r_wp = wp.array(coords_r, dtype=wp.vec3i, device=a.device)
    volume_r = wp.Volume.allocate_by_voxels(
        voxel_points=coords_r_wp, voxel_size=a.dx, device=a.device,
    )
    coords_canonical = volume_r.get_voxels()
    n_active = int(coords_canonical.shape[0])
    values_r = wp.empty(n_active, dtype=float, device=a.device)

    # Background-sign args (defaults if either input lacks one)
    bg_a = a.bg_sign if a.bg_sign is not None else wp.zeros((1, 1, 1), dtype=wp.int32, device=a.device)
    bg_b = b.bg_sign if b.bg_sign is not None else wp.zeros((1, 1, 1), dtype=wp.int32, device=a.device)
    leaf = a.bg_leaf_size

    wp.launch(
        _boolean_combine_kernel,
        dim=n_active,
        inputs=[
            a.volume.id, b.volume.id,
            bg_a,
            a.bg_origin[0], a.bg_origin[1], a.bg_origin[2],
            bg_a.shape[0], bg_a.shape[1], bg_a.shape[2],
            bg_b,
            b.bg_origin[0], b.bg_origin[1], b.bg_origin[2],
            bg_b.shape[0], bg_b.shape[1], bg_b.shape[2],
            leaf,
            coords_canonical,
            a.values, b.values, values_r,
            op, band,
        ],
        device=a.device,
    )
    wp.synchronize()

    # Combine background signs for the result
    bg_r_np, origin_r = _combine_bg_sign(a, b, op)
    bg_r = wp.array(bg_r_np.astype(np.int32), dtype=wp.int32, device=a.device) if bg_r_np is not None else None

    result = SparseSDF(volume_r, coords_canonical, values_r, a.dx, band, a.device,
                       bg_sign=bg_r, bg_origin=origin_r, bg_leaf_size=leaf)
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


def difference(a: SparseSDF, b: SparseSDF, margin: int = 1,
               auto_reinit: int = 3) -> SparseSDF:
    """A − B. Result topology is A.coords ∪ B.coords (B's contribution is
    needed where A's deep interior meets B's surface)."""
    return _apply_boolean(a, b, op=2, margin=margin, auto_reinit=auto_reinit)
