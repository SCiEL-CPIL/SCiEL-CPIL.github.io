"""Heilbronn-11: maximize min triangle area over 11 points in equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145  # safety margin from 150s limit

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165


def _project(pts):
    """Project points into the equilateral triangle (0,0),(1,0),(0.5,S3H)."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_area(pts):
    """Hard minimum triangle area (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    """Differentiable soft-min of triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = np.abs(cross) * 0.5
    # Use log-sum-exp for numerical stability
    a_max = np.max(areas)
    shifted = -beta * (areas - a_max)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return a_max - log_sum / beta


def _hard_min_neg(x):
    """Negative of hard min area for Nelder-Mead (direct, non-smooth)."""
    p = x.reshape(-1, 2)
    p = _project(p)
    return -np.min(_min_area(p))


def _soft_min_neg(x, beta):
    """Negative of soft min area for L-BFGS-B."""
    p = x.reshape(-1, 2)
    p = _project(p)
    return -_soft_min_area(p, beta)


def _good_init(rng, scale=0.0):
    """Symmetric-ish initialization: corners, edge midpoints, interior points."""
    # Start from a reasonable symmetric configuration
    base = np.array([
        [0.0, 0.0],           # corner 0
        [1.0, 0.0],           # corner 1
        [0.5, S3H],           # corner 2
        [0.5, 0.0],           # mid bottom
        [0.75, S3H / 2],      # mid right
        [0.25, S3H / 2],      # mid left
        [1/3, S3H/3],         # centroid
        [1/3, S3H/6],         # lower interior
        [2/3, S3H/6],         # lower interior
        [0.5, S3H/2],         # upper interior
        [0.5, S3H/3],         # center-ish
    ], dtype=float)
    if scale > 0:
        base = base + rng.normal(0, scale, size=(N, 2))
    return _project(base)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project(pts)
        area = np.min(_min_area(pts))
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Phase 1: Multi-start L-BFGS-B with annealed soft-min ----
    n_starts = 20
    scales = [0.0, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        scale = scales[s % len(scales)]
        pts0 = _good_init(rng, scale)
        _update_best(pts0)
        
        # Annealed soft-min optimization
        pts = pts0.copy()
        for beta in [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            if not _elapsed_ok(20):
                break
            try:
                x0 = pts.flatten()
                res = minimize(_soft_min_neg, x0, args=(beta,),
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14})
                pts = _project(res.x.reshape(-1, 2))
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase 2: Hard-min Nelder-Mead polish on best configuration ----
    if best_pts is not None and _elapsed_ok(15):
        # Multiple Nelder-Mead restarts from slightly perturbed best
        for attempt in range(6):
            if not _elapsed_ok(10):
                break
            pert_scale = 0.01 * (1 - attempt * 0.15)
            if pert_scale < 0.001:
                pert_scale = 0.001
            x0 = (best_pts + rng.normal(0, pert_scale, size=(N, 2))).flatten()
            try:
                res = minimize(_hard_min_neg, x0, method='Nelder-Mead',
                              options={'maxiter': 30000, 'xatol': 1e-12, 'fatol': 1e-14,
                                       'adaptive': True})
                p = _project(res.x.reshape(-1, 2))
                _update_best(p)
            except Exception:
                pass

    # ---- Phase 3: Final ultra-fine polish ----
    if best_pts is not None and _elapsed_ok(5):
        try:
            x0 = best_pts.flatten()
            res = minimize(_hard_min_neg, x0, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16,
                                   'adaptive': True})
            p = _project(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # Fallback: random points
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u * 1.0 + v * 0.5
        y = v * S3H
        best_pts = np.column_stack([x, y])
        best_pts = _project(best_pts)

    return _project(best_pts)
