import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0


def _normalize(points):
    pts = np.asarray(points, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_alphas(points):
    """Return per-facet angular circumradius (radians). Returns [] if hull fails."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    try:
        simplices = hull.simplices
    except Exception:
        return []
    if simplices.size == 0:
        return []
    tri = points[simplices]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    nrm = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(nrm, axis=1)
    mask = nn > 1e-12
    if not np.any(mask):
        return []
    d = np.abs(nrm[mask] @ p0[mask].T) / nn[mask]
    d = np.clip(d, 0.0, 1.0)
    return np.arccos(d)


def _covering_radius(points):
    """Exact covering radius in radians (max facet circumradius)."""
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return np.pi
    return float(np.max(alphas))


def _softmax(values, beta):
    if values.size == 0:
        return np.pi
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m))) / beta


def _smooth_obj(params, beta):
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6
    return _softmax(alphas, beta)


def _smooth_obj_grad(params, beta):
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros_like(params)
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6, np.zeros_like(params)
    obj = _softmax(alphas, beta)

    m = np.max(alphas)
    w = np.exp(alphas - m) / (beta * np.sum(np.exp(alphas - m)))

    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        return obj, np.zeros_like(params)

    grad = np.zeros((N, 3))
    for i, simp in enumerate(simplices):
        wi = w[i]
        if wi <= 0.0:
            continue
        a, b, c = pts[simp]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        n = nrm / nn
        d = np.clip(np.abs(n @ a), 0.0, 1.0)
        denom = np.sqrt(max(1.0 - d * d, 1e-14))
        s = wi / denom
        grad[a is a]  # no-op guard
        ga = -s * n
        gb = -s * n
        gc = -s * n
        grad[simp[0]] += ga
        grad[simp[1]] += gb
        grad[simp[2]] += gc
    return obj, grad.flatten()


def _repulsive_relaxation(points, steps=300, alpha=0.05, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    pts = points.copy()
    for _ in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        inv_d3 = 1.0 / (d2 * np.sqrt(d2))
        forces = np.einsum('ij,ijk->ik', inv_d3, diff)
        pts = _normalize(pts + alpha * forces)
    return pts


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    def remaining():
        return deadline - time.time()

    best = _fibonacci()
    best_r = _covering_radius(best)

    def time_ok(budget=0.5):
        return remaining() > budget

    def local_optimize(pts, budget):
        nonlocal best, best_r
        pts = _normalize(pts)
        r0 = _covering_radius(pts)
        if r0 < best_r:
            best = pts.copy()
            best_r = r0

        # Phase A: gradient-based on smooth surrogate (coarse -> fine)
        for beta, maxiter in ((25.0, 300), (40.0, 400), (60.0, 500)):
            if not time_ok(2.0) or (time.time() - start) > budget:
                break
            try:
                res = minimize(_smooth_obj, pts.flatten(), method='L-BFGS-B',
                               jac=lambda x: _smooth_obj_grad(x, beta)[1],
                               options={'maxiter': maxiter, 'gtol': 1e-8,
                                        'ftol': 1e-10, 'maxfun': 2000, 'disp': False})
                if np.all(np.isfinite(res.x)):
                    cand = _normalize(res.x.reshape(N, 3))
                    rc = _covering_radius(cand)
                    if rc < best_r:
                        best = cand
                        best_r = rc
                    if rc < r0:
                        r0 = rc
                        pts = cand
            except Exception:
                pass

        # Phase B: Nelder-Mead polish with exact objective
        if time_ok(2.0) and (time.time() - start) < budget:
            try:
                res = minimize(_covering_radius, best.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                if np.all(np.isfinite(res.x)) and res.fun < best_r:
                    best = _normalize(res.x.reshape(N, 3))
                    best_r = float(res.fun)
            except Exception:
                pass
        return best_r

    # 1. Initial relaxation + first optimization
    if time_ok(1.0):
        rel = _repulsive_relaxation(best, steps=1500, alpha=0.04,
                                    time_limit=6.0, start_time=start)
        rr = _covering_radius(rel)
        if rr < best_r:
            best = rel
            best_r = rr
    local_optimize(best, budget=25.0)

    # 2. Adaptive random restarts
    stall = 0
    scale = 0.45
    while time_ok(12.0):
        noise = np.random.normal(0.0, scale, size=best.shape)
        pert = _normalize(best + noise)
        rel = _repulsive_relaxation(pert, steps=250, alpha=0.05,
                                    time_limit=2.5, start_time=start)
        rr = _covering_radius(rel)
        if rr < best_r:
            best = rel
            best_r = rr
            stall = 0
            scale = max(0.08, scale * 0.85)
            local_optimize(best, budget=12.0)
        else:
            stall += 1
            scale = min(1.0, scale * 1.15)
            if stall >= 12:
                break

    # 3. Final tight polish
    if time_ok(5.0):
        try:
            res = minimize(_covering_radius, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_r:
                best = _normalize(res.x.reshape(N, 3))
                best_r = float(res.fun)
        except Exception:
            pass

    return best
