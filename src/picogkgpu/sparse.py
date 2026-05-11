"""SparseSDF: signed distance field on a wp.Volume topology + flat value buffer.

The natural unit for Phase 1.3+ work. Owns:
  - a wp.Volume (NanoVDB topology) for coord->index lookups in kernels
  - a wp.array1d[float] of active values, aligned with the volume's iteration
  - the spacing (dx) and band half-width (band_width) used to build it

Construct via `SparseSDF.from_dense(phi, dx, band_width)`. Methods on the class
are pure: each returns a new SparseSDF; the underlying buffers are never
mutated in place (this keeps Warp graph capture safe and makes the
implementation match the autodiff path we'll want in Phase 6).

The reinit step kernel lives in `picogkgpu.sparse_reinit` and is reused here;
boolean ops live in `picogkgpu.booleans` and take SparseSDF inputs.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from picogkgpu.sparse_reinit import hj_weno5_reinit_step_sparse, BG_DETECT


BG_LEAF_SIZE = 8  # background-sign field is one entry per 8x8x8 dense block


class SparseSDF:
    def __init__(
        self,
        volume: wp.Volume,
        coords: wp.array,
        values: wp.array,
        dx: float,
        band_width: float,
        device: str,
        bg_sign: wp.array | None = None,
        bg_origin: tuple[int, int, int] = (0, 0, 0),
        bg_leaf_size: int = BG_LEAF_SIZE,
    ) -> None:
        self.volume = volume
        self.coords = coords       # wp.array (N, 3) int32
        self.values = values       # wp.array (N,) float
        self.dx = float(dx)
        self.band_width = float(band_width)
        self.device = device
        # Background sign: coarse 3D int8 array, one entry per `bg_leaf_size`^3
        # block of the dense grid. Values: -1 = block is entirely inside the
        # solid, +1 = entirely outside, 0 = boundary (block straddles the
        # surface and may contain band cells).
        # `bg_origin` is the (i,j,k) offset of bg_sign[0,0,0] in dense-coord
        # space, allowing the sign field to cover a sub-region of the world.
        self.bg_sign = bg_sign
        self.bg_origin = tuple(int(v) for v in bg_origin)
        self.bg_leaf_size = int(bg_leaf_size)

    @property
    def n_active(self) -> int:
        return int(self.coords.shape[0])

    # ─────────────────────── factories ───────────────────────

    @classmethod
    def from_dense(cls, phi_dense: np.ndarray, dx: float, band_width: float,
                   device: str = "cuda", bg_leaf_size: int = BG_LEAF_SIZE) -> "SparseSDF":
        if phi_dense.ndim != 3:
            raise ValueError("phi_dense must be a 3D array")
        mask = np.abs(phi_dense) < band_width
        if not mask.any():
            raise ValueError("no active cells with |phi| < band_width")
        coords_in = np.argwhere(mask).astype(np.int32)
        coords_in_wp = wp.array(coords_in, dtype=wp.vec3i, device=device)
        volume = wp.Volume.allocate_by_voxels(
            voxel_points=coords_in_wp, voxel_size=dx, device=device,
        )
        coords = volume.get_voxels()
        coords_np = coords.numpy()
        vals = phi_dense[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]].astype(np.float32)
        out_of_band = np.abs(vals) >= band_width
        vals = np.where(out_of_band, np.sign(vals).astype(np.float32) * band_width, vals)
        values = wp.array(vals.astype(np.float32), dtype=float, device=device)

        bg_sign_np = compute_bg_sign(phi_dense, leaf_size=bg_leaf_size).astype(np.int32)
        bg_sign = wp.array(bg_sign_np, dtype=wp.int32, device=device)
        return cls(volume, coords, values, dx, band_width, device,
                   bg_sign=bg_sign, bg_origin=(0, 0, 0),
                   bg_leaf_size=bg_leaf_size)

    @classmethod
    def from_topology(cls, coords_np: np.ndarray, values_np: np.ndarray,
                      dx: float, band_width: float,
                      device: str = "cuda") -> "SparseSDF":
        """Build directly from a coord list + parallel value array.
        Used by boolean ops once the result topology is known.
        """
        if coords_np.shape[0] != values_np.shape[0]:
            raise ValueError("coords and values must have same length")
        coords_in_wp = wp.array(coords_np.astype(np.int32), dtype=wp.vec3i, device=device)
        volume = wp.Volume.allocate_by_voxels(
            voxel_points=coords_in_wp, voxel_size=dx, device=device,
        )
        # Volume re-sorts coords into its canonical order; we need to map
        # `values_np` from input order to canonical order. A small hash map
        # keyed by (i,j,k) does it.
        canonical = volume.get_voxels().numpy()
        # Build {tuple(coord): input_idx} map
        idx_map = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(coords_np)}
        reordered = np.empty(canonical.shape[0], dtype=np.float32)
        for j, c in enumerate(canonical):
            key = (int(c[0]), int(c[1]), int(c[2]))
            reordered[j] = values_np[idx_map[key]] if key in idx_map else float(band_width)
        # clip out-of-band values defensively
        out_of_band = np.abs(reordered) >= band_width
        reordered = np.where(out_of_band, np.sign(reordered).astype(np.float32) * band_width, reordered)
        values = wp.array(reordered.astype(np.float32), dtype=float, device=device)
        coords = volume.get_voxels()
        return cls(volume, coords, values, dx, band_width, device)

    # ─────────────────────── conversions ───────────────────────

    def to_dense(self, shape: tuple[int, int, int], bg_value: float | None = None) -> np.ndarray:
        if bg_value is None:
            bg_value = self.band_width
        out = np.full(shape, bg_value, dtype=np.float32)
        coords_np = self.coords.numpy()
        vals_np = self.values.numpy()
        # clamp coords to the requested shape (defensive)
        in_bounds = (
            (coords_np[:, 0] < shape[0]) & (coords_np[:, 1] < shape[1])
            & (coords_np[:, 2] < shape[2])
            & (coords_np[:, 0] >= 0) & (coords_np[:, 1] >= 0) & (coords_np[:, 2] >= 0)
        )
        c = coords_np[in_bounds]
        v = vals_np[in_bounds]
        out[c[:, 0], c[:, 1], c[:, 2]] = v
        return out

    # ─────────────────────── ops ───────────────────────

    def clone(self) -> "SparseSDF":
        new_values = wp.empty_like(self.values)
        wp.copy(new_values, self.values)
        return SparseSDF(self.volume, self.coords, new_values,
                         self.dx, self.band_width, self.device,
                         bg_sign=self.bg_sign, bg_origin=self.bg_origin,
                         bg_leaf_size=self.bg_leaf_size)

    def reinit(self, num_steps: int, dt: float | None = None) -> "SparseSDF":
        """N steps of HJ-WENO5 re-distancing. Returns a new SparseSDF."""
        if num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if num_steps == 0:
            return self.clone()
        dt_eff = float(dt) if dt is not None else 0.5 * self.dx
        phi_a = wp.clone(self.values)
        phi_b = wp.empty_like(phi_a)
        phi0 = wp.clone(self.values)
        a, b = phi_a, phi_b
        for _ in range(num_steps):
            wp.launch(
                hj_weno5_reinit_step_sparse,
                dim=self.n_active,
                inputs=[self.volume.id, self.coords, a, phi0, b,
                        self.dx, dt_eff, self.band_width, BG_DETECT],
                device=self.device,
            )
            a, b = b, a
        wp.synchronize()
        return SparseSDF(self.volume, self.coords, a,
                         self.dx, self.band_width, self.device,
                         bg_sign=self.bg_sign, bg_origin=self.bg_origin,
                         bg_leaf_size=self.bg_leaf_size)


def compute_bg_sign(phi_dense: np.ndarray, leaf_size: int = BG_LEAF_SIZE,
                    tol: float = 1e-6) -> np.ndarray:
    """One int8 per leaf_size^3 block: -1 (all inside), +1 (all outside), 0 (boundary).

    Used by boolean kernels to assign a correct sign to cells that are outside
    a SparseSDF's narrow band — the band-extrapolation fallback `+band_width`
    alone is wrong for cells deep inside thick interiors.

    Implementation: for each leaf-sized block, check the min and max of phi.
    If both negative (with a small tolerance), the block is interior; both
    positive, exterior; otherwise the block straddles the zero contour and
    we return 0 — meaning "use the band-edge fallback at this resolution."
    """
    n0, n1, n2 = phi_dense.shape
    nl0 = (n0 + leaf_size - 1) // leaf_size
    nl1 = (n1 + leaf_size - 1) // leaf_size
    nl2 = (n2 + leaf_size - 1) // leaf_size
    bg = np.zeros((nl0, nl1, nl2), dtype=np.int8)
    for li in range(nl0):
        i0, i1 = li * leaf_size, min((li + 1) * leaf_size, n0)
        for lj in range(nl1):
            j0, j1 = lj * leaf_size, min((lj + 1) * leaf_size, n1)
            for lk in range(nl2):
                k0, k1 = lk * leaf_size, min((lk + 1) * leaf_size, n2)
                block = phi_dense[i0:i1, j0:j1, k0:k1]
                lo = block.min()
                hi = block.max()
                if hi < -tol:
                    bg[li, lj, lk] = -1
                elif lo > tol:
                    bg[li, lj, lk] = +1
                # else: boundary, stays 0
    return bg
