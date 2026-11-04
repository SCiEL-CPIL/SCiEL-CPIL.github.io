import numpy as np
import math
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0

_SQRT3 = np.sqrt(3.0)
_HEX_ANGLES = np.radians(np.array([0, 60, 120, 180, 240, 300]))
_OUTER_NORMAL_ANGLES = np.radians(np.array([30, 90, 150, 210, 270, 330]))


def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = _HEX_ANGLES + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))


def _hex_edge_normals(rot_deg):
    base = _OUTER_NORMAL_ANGLES + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))


def _sat_sep(v1, v2, n1, n2):
    min_overlap = float('inf')
    for nm in (n1, n2):
        for k in range(6):
            n = nm[k]
            p1 = v1 @ n
            p2 = v2 @ n
            mn1 = np.min(p1)
            mx1 = np.max(p1)
            mn2 = np.min(p2)
            mx2 = np.max(p2)
            if mx1 < mn2:
                return mn2 - mx1
            if mx2 < mn1:
                return mn1 - mx2
            ov = min(mx1, mx2) - max(mn1, mn2)
            if ov < min_overlap:
                min_overlap = ov
                if min_overlap <= 0.0:
                    return -min_overlap
    return -min_overlap


def _outer_hex_side_fast(points, outer_rot_deg):
    angles = _OUTER_NORMAL_ANGLES + np.radians(outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / _SQRT3) * max_proj


def _check_disjoint(inner, tol=1e-7):
    n = len(inner)
    verts, norms = [], []
    for i in range(n):
        x, y, rot = inner[i]
        verts.append(_hex_vertices(x, y, 1.0, rot))
        norms.append(_hex_edge_normals(rot))
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_sep(verts[i], verts[j], norms[i], norms[j]) < -tol:
                return False
    return True


def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    spacing_x = _SQRT3 + gap
    spacing_y = 1.5 + 0.5 * gap
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _make_ring(gap=0.0):
    pts = []
    d1 = _SQRT3 + gap
    d2 = 2 * _SQRT3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _make_cluster(gap=0.0):
    """Create a compact cluster of 12 hexagons in a 3x4 arrangement with tight packing."""
    pts = []
    spacing_x = _SQRT3 + gap
    spacing_y = 1.5 + 0.5 * gap
    # 3 rows x 4 cols
    for r in range(3):
        y = r * spacing_y
        for c in range(4):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _make_tight_ring(gap=0.02):
    """Create a ring with very tight gap for better initial packing."""
    pts = []
    d1 = _SQRT3 + gap
    d2 = 2 * _SQRT3 + gap * 0.5
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts, dtype=float)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts


def _compute_penalty(overlaps, beta):
    if len(overlaps) == 0:
        return 0.0
    p_arr = np.array(overlaps)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500.0:
        return float(np.sum(p_arr ** 2))
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return float(beta * lse)


def _objective(p, t0, deadline, beta=50.0):
    if time.time() - t0 > deadline:
        return 1e6

    inner = p[:36].reshape(12, 3)
    ox, oy, oang = p[36], p[37], p[38]

    verts, norms = [], []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        verts.append(v)
        norms.append(_hex_edge_normals(rot))

    overlaps = []
    for i in range(12):
        for j in range(i + 1, 12):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > 12.0:
                continue
            d = _sat_sep(verts[i], verts[j], norms[i], norms[j])
            if d < 0:
                overlaps.append(-d)

    pen = _compute_penalty(overlaps, beta)

    all_pts = np.vstack(verts)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    L = _outer_hex_side_fast(all_pts, oang)
    return L + 100.0 * pen


def _polish(params, t0, deadline):
    elapsed = time.time() - t0
    remaining = deadline - elapsed
    if remaining < 5:
        return params

    # Stage 1: L-BFGS-B with moderate penalty
    if remaining > 15:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 10, 30.0),
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000})
            if res.fun < _objective(params, t0, deadline, 30.0):
                params = res.x
        except Exception:
            pass

    # Stage 2: L-BFGS-B with higher penalty
    if time.time() - t0 < deadline - 8:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 5, 100.0),
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 5000})
            if res.fun < _objective(params, t0, deadline, 100.0):
                params = res.x
        except Exception:
            pass

    # Stage 3: L-BFGS-B with very high penalty for near-exact solution
    if time.time() - t0 < deadline - 3:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 2, 500.0),
                           method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 3000})
            if res.fun < _objective(params, t0, deadline, 500.0):
                params = res.x
        except Exception:
            pass

    # Stage 4: Nelder-Mead final refinement for fine-tuning
    if time.time() - t0 < deadline - 1:
        try:
            res = minimize(_objective, params, args=(t0, deadline - 0.5, 1000.0),
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < _objective(params, t0, deadline, 1000.0):
                params = res.x
        except Exception:
            pass

    return params


def construct():
    t0 = time.time()

    def time_left():
        return DEADLINE - (time.time() - t0)

    configs = []
    
    # Tight grid configurations
    configs.append(_make_hex_grid(4, 3, gap=0.02))
    configs.append(_make_hex_grid(3, 4, gap=0.02))
    configs.append(_make_hex_grid(2, 6, gap=0.02))
    configs.append(_make_hex_grid(6, 2, gap=0.02))
    
    # With rotation offset
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=0.02))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=0.02))
    
    # Ring configurations
    configs.append(_make_ring(gap=0.02))
    configs.append(_make_tight_ring(gap=0.02))
    
    # Cluster configuration
    configs.append(_make_cluster(gap=0.02))
    
    # Perturbed ring
    np.random.seed(42)
    ring_p = _make_ring(gap=0.02).copy()
    ring_p[:, 0] += np.random.randn(12) * 0.1
    ring_p[:, 1] += np.random.randn(12) * 0.1
    configs.append(ring_p)
    
    # Perturbed grid
    np.random.seed(123)
    grid_p = _make_hex_grid(4, 3, gap=0.02).copy()
    grid_p[:, 0] += np.random.randn(12) * 0.08
    grid_p[:, 1] += np.random.randn(12) * 0.08
    configs.append(grid_p)
    
    # Random starts with varied rotations
    for seed in [1, 7, 13, 42]:
        np.random.seed(seed)
        pts = np.random.randn(12, 3) * 2.0
        pts[:, 2] = np.random.choice([0, 30, 15, 45, 60], size=12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)

    best_L = float('inf')
    best_params = None
    outer_rots = [0.0, 30.0, 15.0, 45.0, 22.5, 67.5]

    for cfg in configs:
        if time_left() < 12:
            break
        for orot in outer_rots:
            if time_left() < 6:
                break
            init = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))

            polished = _polish(init, t0, DEADLINE)

            inner = polished[:36].reshape(12, 3)
            ox, oy, oang = polished[36], polished[37], polished[38]
            verts = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(12)]
            norms = [_hex_edge_normals(inner[i, 2]) for i in range(12)]

            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if _sat_sep(verts[i], verts[j], norms[i], norms[j]) < -1e-7:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                all_pts = np.vstack(verts)
                all_pts[:, 0] -= ox
                all_pts[:, 1] -= oy
                L = _outer_hex_side_fast(all_pts, oang)
                if L < best_L:
                    best_L = L
                    best_params = polished.copy()

    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))

    inner = []
    for i in range(12):
        inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])

    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001

    return inner, outer_center, outer_side, outer_angle
