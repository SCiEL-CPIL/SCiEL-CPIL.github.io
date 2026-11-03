import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_unit_vertices(angle_deg):
    """Vertices of a unit regular hexagon centered at origin with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _cluster_positions(gap=0.004):
    """Compact rigid cluster: center + 6 ring + 4 second-ring, hex lattice, small gap."""
    s = 1.0 + gap
    u = np.array([_SQRT3 * s, 0.0])
    v = np.array([_SQRT3 / 2 * s, 1.5 * s])
    coords = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    coords.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    coords.extend(ring2_sorted[:4])
    pos = []
    for ti, tj in coords[:N]:
        pos.append(ti * u + tj * v)
    return np.array(pos)


def _rotated_cluster(pos, cluster_rot_deg):
    """Rotate cluster points about origin by cluster_rot_deg."""
    a = math.radians(cluster_rot_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return pos @ R.T


def _all_vertices(cluster_pos, inner_angle_deg):
    """All 6*N vertices of the inner hexagons in world coords."""
    base = _hex_unit_vertices(inner_angle_deg)  # (6,2)
    # each hexagon i: cluster_pos[i] + base
    verts = cluster_pos[:, np.newaxis, :] + base[np.newaxis, :, :]
    return verts.reshape(-1, 2)


def _enclosing_hex_side_lp(verts, outer_angle_deg):
    """Exact min side length of regular hexagon (fixed orientation) containing verts.
    Returns (side, center). Convex LP in (cx, cy, t)."""
    base_angle = math.radians(outer_angle_deg + 30.0)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)

    M = len(verts)
    dots = verts @ normals.T  # (M,6)

    A = np.zeros((2 * M * 6, 3))
    b = np.zeros(2 * M * 6)
    idx = 0
    for k in range(6):
        nk = normals[k]
        A[idx:idx + M, 0] = nk[0]
        A[idx:idx + M, 1] = nk[1]
        A[idx:idx + M, 2] = -1.0
        b[idx:idx + M] = dots[:, k]
        idx += M
        A[idx:idx + M, 0] = -nk[0]
        A[idx:idx + M, 1] = -nk[1]
        A[idx:idx + M, 2] = -1.0
        b[idx:idx + M] = -dots[:, k]
        idx += M

    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            side = res.x[2] * 2.0 / _SQRT3
            return side, (res.x[0], res.x[1])
    except Exception:
        pass
    # Fallback: centroid
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    proj = rel @ normals.T
    r = np.max(np.maximum(np.max(proj, axis=0), -np.min(proj, axis=0)))
    return r * 2.0 / _SQRT3, (cx, cy)


def _objective(params, cluster_pos, inner_angle_deg):
    """params = [cluster_rot_deg, outer_angle_deg]. Exact LP objective."""
    cr = params[0]
    oa = params[1]
    rot = _rotated_cluster(cluster_pos, cr)
    verts = _all_vertices(rot, inner_angle_deg)
    side, _ = _enclosing_hex_side_lp(verts, oa)
    return side


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner_angle = 30.0
    base_cluster = _cluster_positions(gap=0.004)

    best_side = float('inf')
    best_cr = 0.0
    best_oa = 0.0

    # Coarse sweep: cluster rotation (0..30, hex symmetry period is 60 but container
    # also rotates; use 0..30 with 2.5 step) and outer angle (0..30 with 2.5 step)
    cr_grid = list(np.arange(0.0, 30.0 + 1e-9, 2.5))
    oa_grid = list(np.arange(0.0, 30.0 + 1e-9, 2.5))

    for cr in cr_grid:
        if time.time() > _DEADLINE - 30.0:
            break
        rot = _rotated_cluster(base_cluster, cr)
        verts = _all_vertices(rot, inner_angle)
        for oa in oa_grid:
            side, _ = _enclosing_hex_side_lp(verts, oa)
            if side < best_side:
                best_side = side
                best_cr = cr
                best_oa = oa

    # Refine with Nelder-Mead on the exact (LP) objective
    if time.time() < _DEADLINE - 20.0:
        x0 = np.array([best_cr, best_oa])
        try:
            res = minimize(_objective, x0, args=(base_cluster, inner_angle),
                           method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_cr = res.x[0]
                best_oa = res.x[1]
        except Exception:
            pass

    # Final high-precision polish: very tight Nelder-Mead
    if time.time() < _DEADLINE - 8.0:
        x0 = np.array([best_cr, best_oa])
        try:
            res = minimize(_objective, x0, args=(base_cluster, inner_angle),
                           method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_cr = res.x[0]
                best_oa = res.x[1]
        except Exception:
            pass

    # Build final world positions
    rot = _rotated_cluster(base_cluster, best_cr)
    verts = _all_vertices(rot, inner_angle)
    final_side, center = _enclosing_hex_side_lp(verts, best_oa)
    best_side = final_side

    inner = [[float(rot[i, 0]), float(rot[i, 1]), float(inner_angle)] for i in range(N)]
    return inner, (float(center[0]), float(center[1])), float(best_side), float(best_oa)
