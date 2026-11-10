# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 575.0  # 55s safety margin under 630s wall clock


def _hex_verts(cx, cy, d, ang_rad):
    a = ang_rad + np.deg2rad(np.arange(6) * 60.0)
    return np.stack([cx + d * np.cos(a), cy + d * np.sin(a)], axis=1)


# Precomputed unit-hexagon (side 1, angle 0) data
_UNIT_VERTS = _hex_verts(0.0, 0.0, 1.0, 0.0)                       # (6,2)
_UNIT_EDGE_VECTORS = np.roll(_UNIT_VERTS, -1, axis=0) - _UNIT_VERTS  # (6,2)
_UNIT_EDGE_NORMALS = np.stack([-_UNIT_EDGE_VECTORS[:, 1], _UNIT_EDGE_VECTORS[:, 0]], axis=1)
_UNIT_EDGE_NORMALS /= np.linalg.norm(_UNIT_EDGE_NORMALS, axis=1, keepdims=True)
_UNIT_PROJ = _UNIT_VERTS @ _UNIT_EDGE_NORMALS.T                     # (6,6)
_UNIT_PMIN = _UNIT_PROJ.min(axis=0)
_UNIT_PMAX = _UNIT_PROJ.max(axis=0)
_UNIT_CROSS = np.cross(_UNIT_EDGE_VECTORS, _UNIT_VERTS)             # (6,)

# All 31 distinct hexagonal topologies (centered, sorted)
_TOPOLOGIES = [
    [(-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (-1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), (-2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (-2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (0, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)],
    [(-2, -1), (-1, -1), (0, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (1, 2), (2, 2)],
]

_A1 = np.array([2.0, 0.0])
_A2 = np.array([1.0, np.sqrt(3.0)])
_EPS = 1e-8


def _all_centers_valid(centers):
    """Vectorized SAT: every pair of unit hexagons must be separated by > 1e-6."""
    n = centers.shape[0]
    D = centers[None, :, :] - centers[:, None, :]              # (n, n, 2)
    proj = D @ _UNIT_EDGE_NORMALS.T                             # (n, n, 6)
    lo = proj + _UNIT_PMIN[None, None, :]
    hi = proj + _UNIT_PMAX[None, None, :]
    gap = np.maximum(hi - lo, lo - hi)                          # (n, n, 6)
    m = np.arange(n)
    return bool(np.all(gap[np.triu_indices(n, k=1)] > 1e-6))


def _min_s_for_angle(centers, ang_rad):
    """Minimum side length of outer hexagon (centered at origin, angle ang_rad) containing all inner hexagons."""
    n = centers.shape[0]
    V = _UNIT_VERTS[None, None, :, :] + centers[None, :, :, None]   # (n, 6, 2)
    ang = ang_rad + np.deg2rad(np.arange(6) * 60.0)
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (6, 2)
    dots = np.einsum('nvi,ji->njv', V, normals)                     # (n, 6, 6)
    return float(np.max(dots) / (np.sqrt(3.0) / 2.0))


def _outer_contains(centers, s, ang_rad):
    """Check all inner vertices lie inside outer hexagon (cross-product test)."""
    n = centers.shape[0]
    V = _UNIT_VERTS[None, :, :, :] + centers[:, None, :]            # (n, 6, 2)
    ov = s * _hex_verts(0.0, 0.0, 1.0, ang_rad)                     # (6, 2)
    oe = np.roll(ov, -1, axis=0) - ov
    cross = oe[:, 0] * (V[..., 1] - ov[None, None, :, 1]) - oe[:, 1] * (V[..., 0] - ov[None, None, :, 0])
    return bool(np.all(cross >= -1e-6))


def _local_refine(centers, ang_rad, s, max_iter=300):
    """Nudge-shift hexagons toward the outer boundary to reduce required s."""
    centers = centers.copy()
    n = centers.shape[0]
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        s = _min_s_for_angle(centers, ang_rad)
        if s < np.sqrt(12.0) - 1e-4:
            break
        ov = s * _hex_verts(0.0, 0.0, 1.0, ang_rad)
        oe = np.roll(ov, -1, axis=0) - ov
        on = np.stack([-oe[:, 1], oe[:, 0]], axis=1)
        on /= np.linalg.norm(on, axis=1, keepdims=True)
        # For each inner hex, find the outer edge it's closest to, and nudge it outward
        improved = False
        for i in range(n):
            vi = _UNIT_VERTS + centers[i]
            dists = np.min(np.cross(oe, vi[None, :] - ov[:, None, :]), axis=1)
            k = int(np.argmin(dists))
            step = 0.0005
            new_c = centers[i] + step * on[k]
            trial = centers.copy()
            trial[i] = new_c
            if _all_centers_valid(trial):
                s2 = _min_s_for_angle(trial, ang_rad)
                if s2 < s - 1e-9:
                    centers = trial
                    s = s2
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    s = _min_s_for_angle(centers, ang_rad)
    if _all_centers_valid(centers) and _outer_contains(centers, s, ang_rad):
        return centers, s
    return None, None


def hexagon_packing_12():
    best_s = float('inf')
    best_centers = None
    best_ang = 0.0

    # Precompute centers for all topologies
    topo_centers = []
    for topo in _TOPOLOGIES:
        c = np.array(topo, dtype=float)
        c = c @ np.array([_A1, _A2]).T
        c -= c.mean(axis=0)
        topo_centers.append(c)

    # Coarse angle sweep: 0..60 in 1-degree steps
    coarse_angles = np.deg2rad(np.arange(0, 61, 1))
    for c in topo_centers:
        if time.time() > _DEADLINE:
            break
        if not _all_centers_valid(c):
            continue
        s_coarse = np.array([_min_s_for_angle(c, a) for a in coarse_angles])
        k = int(np.argmin(s_coarse))
        best_ang = float(coarse_angles[k])
        best_s = float(s_coarse[k])
        best_centers = c.copy()

    # Fine angle sweep around best angle (±2° in 0.1° steps)
    fine_angles = np.deg2rad(np.arange(best_ang - 2.0, best_ang + 2.01, 0.1))
    for c in topo_centers:
        if time.time() > _DEADLINE:
            break
        if not _all_centers_valid(c):
            continue
        s_fine = np.array([_min_s_for_angle(c, a) for a in fine_angles])
        k = int(np.argmin(s_fine))
        if s_fine[k] < best_s - 1e-9:
            best_s = float(s_fine[k])
            best_ang = float(fine_angles[k])
            best_centers = c.copy()

    # Local refinement on the best configuration
    if best_centers is not None and time.time() < _DEADLINE - 10:
        ref_c, ref_s = _local_refine(best_centers, best_ang, best_s)
        if ref_c is not None and ref_s < best_s:
            best_centers = ref_c
            best_s = ref_s

    # Final verification and fallback
    if best_centers is None:
        # Fallback: simple 4x3 grid
        coords = [(i, j) for j in range(3) for i in range(4)]
        best_centers = np.array(coords, dtype=float) @ np.array([_A1, _A2]).T
        best_centers -= best_centers.mean(axis=0)
        best_s = _min_s_for_angle(best_centers, 0.0)
        best_ang = 0.0

    # Ensure validity
    if not _all_centers_valid(best_centers):
        # Scale up slightly to ensure separation
        scale = 1.001
        best_centers = best_centers * scale
        best_s = _min_s_for_angle(best_centers, best_ang)

    # Build output
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_centers[i, 0]
        inner[i, 1] = best_centers[i, 1]
        inner[i, 2] = 0.0

    outer = np.array([0.0, 0.0, float(np.rad2deg(best_ang))])
    return inner, outer, float(best_s)


# EVOLVE-BLOCK-END
