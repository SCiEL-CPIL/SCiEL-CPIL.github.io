"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _penalized_soft_min(flat_params, beta):
    """Objective for L-BFGS-B: negative soft-min area with penalty for out-of-triangle."""
    points = flat_params.reshape(-1, 2)
    
    # Compute penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    
    ma = _soft_min_area(points, beta)
    return -ma + penalty


def _penalized_hard_min(flat_params):
    """Objective for Nelder-Mead: negative hard-min area with penalty for out-of-triangle."""
    points = flat_params.reshape(-1, 2)
    
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += abs(y) * 1000
        if y > S3H:
            penalty += (y - S3H) * 1000
        if x < 0:
            penalty += abs(x) * 1000
        if x > 1:
            penalty += (x - 1) * 1000
        left_viol = y - S3 * x
        if left_viol > 0:
            penalty += left_viol * 1000
        right_viol = y + S3 * x - S3
        if right_viol > 0:
            penalty += right_viol * 1000
    
    ma = _min_triangle_area(points)
    return -ma + penalty


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        # Project to ensure validity before evaluating
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial configuration: symmetric placement ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        a = 1 - u - v
        x = a * 0 + u * 1 + v * 0.5
        y = a * 0 + u * 0 + v * S3H
        return np.array([x, y])

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        pts = _project_to_triangle(pts)
        return pts

    # ---- Phase A: Differential Evolution for global exploration ----
    if _elapsed_ok(60):
        try:
            bounds = [(0, 1), (0, S3H)] * N
            de_result = differential_evolution(
                _penalized_hard_min,
                bounds,
                maxiter=30,
                popsize=12,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_result.fun < 0:
                de_pts = de_result.x.reshape(-1, 2)
                de_pts = _project_to_triangle(de_pts)
                _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Multi-start soft-min gradient optimization (annealed beta) ----
    n_starts = 18
    bases = [init.copy()]
    for _ in range(10):
        bases.append(_start_from_base(init, 0.25))
    if best_pts is not None:
        bases.append(best_pts.copy())

    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.06)
        _update_best(pts)
        
        for beta in [12.0, 35.0, 90.0, 220.0, 500.0, 1000.0]:
            if not _elapsed_ok(35):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _penalized_soft_min(x, b),
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                pts = res.x.reshape(-1, 2)
                pts = _project_to_triangle(pts)
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase C: Nelder-Mead local polish on the best configuration ----
    if best_pts is not None and _elapsed_ok(15):
        for scale in [0.05, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(5):
                if not _elapsed_ok(8):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                
                try:
                    res = minimize(
                        _penalized_hard_min,
                        cand.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    p = res.x.reshape(-1, 2)
                    p = _project_to_triangle(p)
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(5):
        try:
            res = minimize(
                _penalized_hard_min,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # fallback random feasible configuration
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    # Final projection to ensure validity
    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
