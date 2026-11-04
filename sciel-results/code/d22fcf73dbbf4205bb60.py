"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0
TRI_AREA = S3 / 4.0

_IDX_TRIPLES = np.array(
    [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
)
_I = _IDX_TRIPLES[:, 0]
_J = _IDX_TRIPLES[:, 1]
_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = _IDX_TRIPLES.shape[0]


def _bary_to_cart(bc):
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * H
    return np.column_stack([x, y])


def _cart_to_bary(points):
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    return np.column_stack([b, c])


def _project_to_bary_simplex(bc):
    b = bc[:, 0].astype(float).copy()
    c = bc[:, 1].astype(float).copy()
    b = np.clip(b, 0.0, None)
    c = np.clip(c, 0.0, None)
    excess = b + c - 1.0
    mask = excess > 0
    if np.any(mask):
        s = 1.0 / (1.0 + excess[mask])
        b[mask] *= s
        c[mask] *= s
    return np.column_stack([b, c])


def _clip_to_triangle(points):
    x = points[:, 0].astype(float).copy()
    y = points[:, 1].astype(float).copy()
    y = np.clip(y, 0.0, H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    return np.column_stack([x, y])


def _dets_from_bary(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui, vi, wi = u[_I], v[_I], w[_I]
    uj, vj, wj = u[_J], v[_J], w[_J]
    uk, vk, wk = u[_K], v[_K], w[_K]
    det = ui * (vj * wk - wj * vk) - vi * (uj * wk - wj * uk) + wi * (uj * vk - vj * uk)
    return det


def _min_area_ratio_from_bary(bary):
    dets = _dets_from_bary(bary)
    return float(np.min(np.abs(dets)))


def _structured_init_fps(rng, n_init):
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    for _ in range(n_init - 3):
        best_val = -1.0
        best_cand = None
        for t in range(3000):
            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            cand = np.array([u, v])
            d2 = np.min(np.sum((np.array(pts) - cand) ** 2, axis=1))
            if d2 > best_val:
                best_val = d2
                best_cand = cand
        pts.append(best_cand)
    return np.array(pts, dtype=float)


def _local_optimize_nm(points, maxiter=3000, deadline=None):
    """Nelder-Mead optimization in Cartesian coordinates."""
    def obj(x):
        pts = _clip_to_triangle(x.reshape(N, 2))
        return -_min_area_ratio_from_bary(_cart_to_bary(pts))
    
    x0 = points.flatten()
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-12,
                                'fatol': 1e-14, 'adaptive': True})
        return _clip_to_triangle(res.x.reshape(N, 2))
    except Exception:
        return points


def _local_optimize_slsqp(points, maxiter=400):
    """SLSQP optimization in barycentric coordinates."""
    bary = _cart_to_bary(points)
    x0 = bary.flatten()
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2 * i] - x[2 * i + 1]})
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def obj_slsqp(x):
        bary_proj = _project_to_bary_simplex(x.reshape(N, 2))
        dets = _dets_from_bary(bary_proj)
        return -float(np.min(np.abs(dets)))
    
    try:
        res = minimize(obj_slsqp, x0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        bary_opt = _project_to_bary_simplex(res.x.reshape(N, 2))
        return _clip_to_triangle(_bary_to_cart(bary_opt))
    except Exception:
        return points


def _final_polish(points, deadline):
    """High-precision final polish using multiple methods."""
    best = points.copy()
    best_score = _min_area_ratio_from_bary(_cart_to_bary(points))
    
    def consider(pts):
        nonlocal best, best_score
        pts = _clip_to_triangle(pts)
        score = _min_area_ratio_from_bary(_cart_to_bary(pts))
        if score > best_score:
            best_score = score
            best = pts.copy()
    
    # SLSQP with high precision
    if time.time() < deadline - 5.0:
        polished = _local_optimize_slsqp(best, maxiter=500)
        consider(polished)
    
    # NM with high precision
    if time.time() < deadline - 3.0:
        polished = _local_optimize_nm(best, maxiter=2000)
        consider(polished)
    
    # Another SLSQP round
    if time.time() < deadline - 1.0:
        polished = _local_optimize_slsqp(best, maxiter=300)
        consider(polished)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -float('inf')
    
    def consider(points):
        nonlocal best_points, best_score
        points = _clip_to_triangle(points)
        score = _min_area_ratio_from_bary(_cart_to_bary(points))
        if score > best_score:
            best_score = score
            best_points = points.copy()
        return score
    
    # Phase 1: Multi-start with Nelder-Mead as primary optimizer
    n_starts = 6
    for s in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        bary_init = _structured_init_fps(rng, N)
        points_init = _bary_to_cart(bary_init)
        points_init = _clip_to_triangle(points_init)
        consider(points_init)
        
        # Nelder-Mead refinement (primary optimizer)
        if time.time() > deadline - 25.0:
            break
        pts_nm = _local_optimize_nm(points_init, maxiter=3000)
        consider(pts_nm)
        
        # SLSQP polish
        if time.time() > deadline - 20.0:
            break
        pts_slsqp = _local_optimize_slsqp(pts_nm, maxiter=400)
        consider(pts_slsqp)
    
    # Phase 2: Final high-precision polish on the best solution
    if best_points is not None:
        if time.time() < deadline - 15.0:
            polished = _final_polish(best_points, deadline)
            consider(polished)
        
        # One more round of NM + SLSQP on the polished result
        if time.time() < deadline - 10.0:
            pts_nm2 = _local_optimize_nm(polished, maxiter=2000)
            consider(pts_nm2)
        
        if time.time() < deadline - 5.0:
            pts_slsqp2 = _local_optimize_slsqp(polished, maxiter=600)
            consider(pts_slsqp2)
        
        if time.time() < deadline - 2.0:
            polished2 = _final_polish(best_points, deadline)
            consider(polished2)
    
    if best_points is None:
        bary_init = _structured_init_fps(rng, N)
        best_points = _bary_to_cart(bary_init)
    
    return _clip_to_triangle(best_points)
