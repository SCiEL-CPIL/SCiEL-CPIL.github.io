"""Structural redesign: 6-fold symmetric triangular-lattice packing of unit hexagons."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
EPS = 1e-9


def _hex_vertices(cx, cy, r, ang):
    a = np.deg2rad(ang)
    th = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def _support(pts, u):
    return float(np.max(pts @ u))


def _min_side_to_contain(pts, ang):
    # outer regular hexagon (flat-top, side s, center 0) contains pts iff
    # s >= max_i support(pts, n_i) where n_i are outward normals of the 6 sides
    a = np.deg2rad(ang)
    th = a + np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    return float(np.max(pts @ normals))


def _overlap_area(ptsA, ptsB):
    n = len(ptsA)
    m = len(ptsB)
    total = 0.0
    for i in range(n):
        A = ptsA[i]
        B = ptsA[(i + 1) % n]
        for j in range(m):
            C = ptsB[j]
            D = ptsB[(j + 1) % m]
            d = B - A
            e = D - C
            denom = d[0] * e[1] - d[1] * e[0]
            if abs(denom) < 1e-12:
                continue
            ac = C - A
            bc = B - C
            t = (ac[0] * e[1] - ac[1] * e[0]) / denom
            u = (ac[0] * d[1] - ac[1] * d[0]) / denom
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                total += t * u
    return total


def _hexagon_overlap(pA, pB, qA, qB, aA, aB):
    A = _hex_vertices(pA[0], pA[1], 1.0, aA)
    B = _hex_vertices(pB[0], pB[1], 1.0, aB)
    return _overlap_area(A, B)


def _lattice_positions(scale, rot):
    # 12 hexagons on a hexagonal (triangular) lattice, 6-fold symmetric
    d = scale
    a = np.deg2rad(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    basis = np.array([
        [d, 0.0],
        [d / 2.0, d * math.sqrt(3) / 2.0],
        [-d / 2.0, d * math.sqrt(3) / 2.0],
        [-d, 0.0],
        [-d / 2.0, -d * math.sqrt(3) / 2.0],
        [d / 2.0, -d * math.sqrt(3) / 2.0],
        [1.5 * d, d * math.sqrt(3) / 2.0],
        [d, d * math.sqrt(3)],
        [-d / 2.0, d * math.sqrt(3)],
        [-d, d * math.sqrt(3) / 2.0],
        [-1.5 * d, -d * math.sqrt(3) / 2.0],
        [-d, -d * math.sqrt(3)],
    ])
    return basis @ R.T


def _objective(x):
    scale, rot, outer_ang = x
    scale = max(scale, 0.5)
    pos = _lattice_positions(scale, rot)
    pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, 0.0) for p in pos])
    side = _min_side_to_contain(pts, outer_ang)
    # pairwise overlap penalty (6-fold symmetry reduces to 18 unique pairs)
    overlap = 0.0
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            ov = _hexagon_overlap(pos[i], pos[j], 0.0, 0.0)
            if ov > 0:
                overlap += ov
    return side + 10.0 * overlap


def _build(pos, outer_ang):
    pos = np.asarray(pos, dtype=float)
    pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, 0.0) for p in pos])
    side = _min_side_to_contain(pts, outer_ang)
    inner = [[float(p[0]), float(p[1]), 0.0] for p in pos]
    return inner, (0.0, 0.0), float(side), float(outer_ang)


def construct():
    t0 = time.time()
    best = None
    best_val = float('inf')

    def consider(x):
        nonlocal best, best_val
        v = _objective(x)
        if v < best_val:
            best_val = v
            best = x

    # coarse grid over (scale, rotation, outer_angle)
    for scale in np.arange(1.5, 3.4, 0.25):
        for rot in np.arange(0.0, 60.0, 15.0):
            for outer_ang in np.arange(0.0, 60.0, 15.0):
                x0 = np.array([scale, rot, outer_ang], dtype=float)
                consider(x0)
                if time.time() - t0 > DEADLINE - 10:
                    break
        if time.time() - t0 > DEADLINE - 10:
            break

    # local Nelder-Mead refinement from the best grid point
    if best is not None and time.time() - t0 < DEADLINE - 15:
        x0 = best.copy()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
        except Exception:
            pass

    # a few extra Nelder-Mead starts from other good grid points
    if time.time() - t0 < DEADLINE - 20:
        starts = [
            [2.0, 0.0, 0.0],
            [2.0, 30.0, 30.0],
            [2.5, 15.0, 0.0],
            [1.8, 0.0, 30.0],
        ]
        for s in starts:
            if time.time() - t0 > DEADLINE - 10:
                break
            try:
                res = minimize(_objective, np.array(s, dtype=float), method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best = res.x
            except Exception:
                pass

    if best is None:
        # fallback: trivial non-overlapping placement
        pos = np.array([[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
        return _build(pos, 0.0)

    scale, rot, outer_ang = best
    pos = _lattice_positions(scale, rot)
    return _build(pos, outer_ang)
