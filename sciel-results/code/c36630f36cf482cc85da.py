"""Rewritten: smooth soft-min covering objective + gradient descent + simulated
annealing + multi-start, then short hull-based facet local refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _project_tangent(p, v):
    """Project v onto tangent plane at unit vector p (removes radial component)."""
    return v - np.dot(v, p) * p

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None
    facets = hull.simplices
    max_circum = 0.0
    worst_facet = None
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n = n / nn
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
        C = n
        cos_R = np.clip(np.dot(C, a), -1.0, 1.0)
        R = np.arccos(cos_R)
        if R > max_circum:
            max_circum = R
            worst_facet = tri
    return np.degrees(max_circum), worst_facet

def _smooth_cover(pts, t):
    """Smooth approximation of the covering radius (radians)."""
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    d = np.arccos(dots)
    np.fill_diagonal(d, 1e9)
    # soft-min: -t * log(mean(exp(-d/t)))
    m = np.min(d, axis=1)
    logsum = m - t * np.log(np.mean(np.exp(-(d - m) / t), axis=1))
    min_d = -logsum  # per-point soft-min distance
    return np.max(min_d)

def _smooth_grad(pts, t):
    """Gradient of the smooth covering objective w.r.t. pts (3D, pre-projection)."""
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    d = np.arccos(dots)
    np.fill_diagonal(d, 1e9)
    m = np.min(d, axis=1)
    expw = np.exp(-(d - m) / t)
    expw[np.eye(N, dtype=bool)] = 0.0
    w = expw / expw.sum(axis=1, keepdims=True)  # softmax weights (per row i over j)
    # d(min_d_i)/d(pts[k])  : min_d_i = -t log sum_j w_j exp(-d_ij/t) ... use softmax form
    # min_d_i = sum_j w_j d_ij  (softmax-weighted average, since w ~ exp(-d/t))
    # d(d_ij)/d(pts[k]) = - (1/sqrt(1-dots^2)) * (pts[i] - pts[k])_k
    # = - (1/sin(d_ij)) * (pts[i] - pts[k])
    sin_d = np.sin(d)
    sin_d = np.where(sin_d < 1e-9, 1e-9, sin_d)
    # G[k] = sum_i w[i,k] * (-1/sin_d[i,k]) * (pts[i] - pts[k])
    coeff = w / sin_d  # [i,k]
    # term: sum_i coeff[i,k] * pts[i]  -> (coeff.T @ pts) [k,3]
    sum_i = coeff.T @ pts  # [k,3]
    # sum_i coeff[i,k]
    sum_c = coeff.sum(axis=0)  # [k]
    grad = sum_i - sum_c[:, None] * pts  # [k,3]
    return grad

def _fibonacci_init(seed):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_init(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def _optimize_smooth(pts, t0, t_min, n_iter, time_budget):
    """Gradient descent + annealing on the smooth objective."""
    t = t0
    lr = 0.05
    for it in range(n_iter):
        if time_budget is not None and time_budget() > 0:
            break
        val = _smooth_cover(pts, t)
        g = _smooth_grad(pts, t)
        # project gradient to tangent space
        g_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
        # step
        pts = pts - lr * g_tan
        pts = _normalize(pts)
        # occasional random kick (annealing)
        if it % 20 == 19 and t > t_min:
            rng = np.random.RandomState(int(time.time()*1000) + it)
            idx = rng.randint(N)
            kick = rng.randn(3)
            kick = _project_tangent(pts[idx], kick)
            kick = kick / (np.linalg.norm(kick) + 1e-12)
            pts[idx] = pts[idx] + 0.05 * kick
            pts = _normalize(pts)
        # anneal
        t = max(t_min, t * 0.995)
        lr = max(0.002, lr * 0.999)
    return pts

def _facet_local_refine(pts, n_rounds, time_budget):
    """Refine worst hull facets with a few smooth gradient steps targeting that facet."""
    for _ in range(n_rounds):
        if time_budget is not None and time_budget() > 0:
            break
        rad, worst = _covering_radius_deg(pts)
        if worst is None:
            break
        tri = worst
        # do a few smooth steps (low t) to reduce the global max
        for _ in range(30):
            if time_budget is not None and time_budget() > 0:
                break
            g = _smooth_grad(pts, 0.02)
            g_tan = g - (g * pts).sum(axis=1, keepdims=True) * pts
            pts = pts - 0.01 * g_tan
            pts = _normalize(pts)
    return pts

def construct():
    t_start = time.time()
    deadline = 150.0
    safety = 12.0

    def remaining():
        return (t_start + deadline - safety) - time.time()

    best_pts = None
    best_score = 1e9

    # Multi-start
    starts = []
    for s in range(6):
        starts.append(_fibonacci_init(s))
    for s in range(4):
        starts.append(_random_init(100 + s))

    for k, init in enumerate(starts):
        if remaining() < 12.0:
            break
        # smooth optimization
        pts = _optimize_smooth(init, t0=0.15, t_min=0.01,
                               n_iter=1200, time_budget=lambda: remaining() < 10.0)
        rad, _ = _covering_radius_deg(pts)
        if rad < best_score:
            best_score = rad
            best_pts = pts.copy()

    # Final hull-based local refinement on the best
    if best_pts is not None and remaining() > 5.0:
        best_pts = _facet_local_refine(best_pts, n_rounds=8, time_budget=lambda: remaining() < 4.0)
        rad, _ = _covering_radius_deg(best_pts)
        if rad < best_score:
            best_score = rad

    if best_pts is None:
        best_pts = _fibonacci_init(0)

    return best_pts
