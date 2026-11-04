"""Heilbronn convex regions: maximize minimum triangle area over convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
M = N * (N - 1) * (N - 2) // 6  # 286 triangles

def _pts_from_params(params):
    """params: (2N,) -> angles (sorted), radii. Returns (N,2)."""
    a = params[:N]
    r = params[N:]
    a = a - a[0]
    a = np.sort(a)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])

def _all_triangle_areas(pts):
    n = len(pts)
    areas = np.empty(M)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas[idx] = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) -
                                       (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                idx += 1
    return areas

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return float(np.min(_all_triangle_areas(pts)) / ha)

def _soft_min_grad(params, T):
    """Return (value, grad) for soft-min objective with analytic gradient."""
    pts = _pts_from_params(params)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6, np.zeros_like(params)
    areas = _all_triangle_areas(pts)
    norm = areas / ha
    # soft min = -T * log(sum(exp(-norm/T)))
    val = -T * np.log(np.sum(np.exp(-norm / T)))
    # gradient w.r.t. norm: d soft_min / d norm_i = exp(-norm_i/T) / sum(exp(-norm_j/T))
    w = np.exp(-norm / T)
    w = w / w.sum()
    # d norm_i / d pts = (1/ha) * d area_i / d pts
    # area_i = 0.5*|cross|; for points in general position (convex, sorted), cross>0
    # d area_i / d p_m = 0.5 * cross(u,v) where (u,v) are the two other points' edges
    # Specifically for triangle (i,j,k) with i<j<k:
    #   d/dp_i = 0.5 * cross(p_j - p_k, ...) -> use standard:
    #   d area/d p_i = 0.5 * ( (p_k - p_j) rotated 90 )
    # Let's use: area = 0.5*(x_j y_k - x_k y_j - x_i y_k + x_k y_i + x_i y_j - x_j y_i)
    # d/dx_i = 0.5*(y_j - y_k), d/dy_i = 0.5*(x_k - x_j)
    # d/dx_j = 0.5*(y_k - y_i), d/dy_j = 0.5*(x_i - x_k)
    # d/dx_k = 0.5*(y_i - y_j), d/dy_k = 0.5*(x_j - x_i)
    dpts = np.zeros((N, 2))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                wi = w[idx] / ha
                if wi > 1e-15:
                    dpts[i, 0] += wi * 0.5 * (pts[j,1] - pts[k,1])
                    dpts[i, 1] += wi * 0.5 * (pts[k,0] - pts[j,0])
                    dpts[j, 0] += wi * 0.5 * (pts[k,1] - pts[i,1])
                    dpts[j, 1] += wi * 0.5 * (pts[i,0] - pts[k,0])
                    dpts[k, 0] += wi * 0.5 * (pts[i,1] - pts[j,1])
                    dpts[k, 1] += wi * 0.5 * (pts[j,0] - pts[i,0])
                idx += 1
    # d pts / d params: pts = r * (cos a, sin a); angles sorted
    a = params[:N]
    r = params[N:]
    grad = np.zeros_like(params)
    for m in range(N):
        grad[m] = -r[m] * np.sin(a[m]) * dpts[m,0] + r[m] * np.cos(a[m]) * dpts[m,1]
        grad[N+m] = np.cos(a[m]) * dpts[m,0] + np.sin(a[m]) * dpts[m,1]
    return val, grad

def _anneal_from(params0, deadline):
    """Anneal temperature from high to low, refining params."""
    params = params0.copy()
    best_val = _soft_min_grad(params, 0.5)[0]
    T = 0.5
    while T > 0.005 and time.time() < deadline:
        for _ in range(2):
            if time.time() > deadline:
                break
            try:
                res = minimize(lambda p: _soft_min_grad(p, T)[0],
                               params,
                               jac=lambda p: _soft_min_grad(p, T)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.x is not None:
                    params = res.x
            except Exception:
                pass
        T *= 0.5
    return params

def construct():
    start = time.time()
    deadline = start + 140.0

    # Initial candidate: regular 13-gon
    angles = 2*np.pi*np.arange(N)/N
    radii = np.ones(N)
    params0 = np.concatenate([angles, radii])
    pts0 = _pts_from_params(params0)
    ha0 = _hull_area(pts0)
    if ha0 > 1e-12:
        params0[N:] = params0[N:] * np.sqrt(1.0/ha0)

    best_pts = _pts_from_params(params0)
    best_score = _score(best_pts)

    # Anneal from regular polygon
    if time.time() < deadline - 10:
        p_annealed = _anneal_from(params0, deadline)
        pts_a = _pts_from_params(p_annealed)
        s = _score(pts_a)
        if s > best_score:
            best_score, best_pts = s, pts_a

    # Halton initial
    def _halton(n, base):
        out = np.zeros(n)
        f = 1.0
        for i in range(1, n+1):
            x = 0.0
            ii = i
            while ii > 0:
                f /= base
                x += f * (ii % base)
                ii //= base
            out[i-1] = x
        return out
    if time.time() < deadline - 15:
        h1 = _halton(N, 2)
        h2 = _halton(N, 3)
        # map to angles and radii
        a_h = 2*np.pi * h1
        r_h = 0.5 + 0.5 * h2
        params_h = np.concatenate([a_h, r_h])
        p_h = _anneal_from(params_h, deadline)
        pts_h = _pts_from_params(p_h)
        s = _score(pts_h)
        if s > best_score:
            best_score, best_pts = s, pts_h

    # Random starts
    rng = np.random.default_rng(42)
    for trial in range(8):
        if time.time() > deadline - 15:
            break
        a_r = np.sort(rng.uniform(0, 2*np.pi, N))
        r_r = rng.uniform(0.5, 1.5, N)
        params_r = np.concatenate([a_r, r_r])
        pts_r = _pts_from_params(params_r)
        ha_r = _hull_area(pts_r)
        if ha_r > 1e-12:
            params_r[N:] = params_r[N:] * np.sqrt(1.0/ha_r)
            p_r = _anneal_from(params_r, deadline)
            pts_r2 = _pts_from_params(p_r)
            s = _score(pts_r2)
            if s > best_score:
                best_score, best_pts = s, pts_r2

    # Final polish with low-T anneal from best
    if time.time() < deadline - 5:
        # recover params from best_pts
        cx, cy = best_pts[:,0].mean(), best_pts[:,1].mean()
        a_b = np.arctan2(best_pts[:,1]-cy, best_pts[:,0]-cx)
        a_b = np.sort(a_b)
        r_b = np.sqrt((best_pts[:,0]-cx)**2 + (best_pts[:,1]-cy)**2)
        params_b = np.concatenate([a_b, r_b])
        # low-T refinement
        for T in [0.01, 0.005, 0.002]:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(lambda p: _soft_min_grad(p, T)[0],
                               params_b,
                               jac=lambda p: _soft_min_grad(p, T)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    params_b = res.x
            except Exception:
                pass
        pts_b = _pts_from_params(params_b)
        s = _score(pts_b)
        if s > best_score:
            best_score, best_pts = s, pts_b

    return best_pts
