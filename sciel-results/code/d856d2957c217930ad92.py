"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Structural change: use a smooth soft-minimum surrogate of the minimum triangle area
combined with the convex hull area in a differentiable objective.  Optimize with
scipy.optimize.minimize (L-BFGS-B) from many diverse starts, then polish the best
result with a short targeted local search that attacks the smallest triangle directly.
This replaces the previous non-gradient simulated annealing + lift family with a
gradient-based global optimization that exploits the smoothness of the surrogate.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
_T0 = time.time()
_DEADLINE = _T0 + 145.0  # wall-clock guard (150 s kill, 5 s safety margin)


def _time_ok(margin=0.0):
    return time.time() < _DEADLINE - margin


def _all_areas(pts):
    """Vectorized triangle areas for all C(n,3) triples."""
    n = len(pts)
    m = n * (n - 1) * (n - 2) // 6
    areas = np.empty(m, dtype=np.float64)
    p = 0
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        mlen = len(rest)
        for j in range(mlen - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            cross = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            a = 0.5 * np.abs(cross)
            cnt = a.size
            areas[p:p + cnt] = a
            p += cnt
    return areas


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    areas = _all_areas(pts)
    if areas.size == 0:
        return 0.0
    return float(np.min(areas)) / hull_a


def _smallest_triangle(pts):
    """Return (min_area, (i, j, k)) for the smallest triangle."""
    n = len(pts)
    best = float('inf')
    bi = bj = bk = 0
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        mlen = len(rest)
        for j in range(mlen - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            cross = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            a = 0.5 * np.abs(cross)
            li = int(np.argmin(a))
            va = a[li]
            if va < best:
                best = va
                bi, bj, bk = i, i + 1 + j, i + 2 + j + li
    return best, (bi, bj, bk)


def _lift(pts, i, j, k, strength=0.02):
    """Move point i directly away from the line through j and k."""
    pts = pts.copy()
    pj, pk, pi = pts[j], pts[k], pts[i]
    d = pk - pj
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        d = pi - pj
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            return pts
        n = d / nrm
        perp = np.array([-n[1], n[0]])
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    else:
        n = d / nrm
        perp = np.array([-n[1], n[0]])
        side = np.sign(np.dot(pi - pj, perp))
        if side == 0:
            side = 1.0
        pts[i] = pi + side * perp * strength * nrm
    return pts


def _soft_min_score(pts, beta=80.0):
    """Differentiable surrogate: -softmin(areas) / hull_area.
    
    softmin(a) = -1/beta * log(sum(exp(-beta*a_i)))
    We minimize -softmin/hull = -softmin * (1/hull)
    """
    areas = _all_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    # Clip to avoid overflow
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    softmin_val = -log_sum / beta
    return -softmin_val / hull_a


def _soft_min_grad(pts, beta=80.0):
    """Gradient of _soft_min_score via finite differences (central)."""
    eps = 1e-6
    grad = np.zeros_like(pts)
    base = _soft_min_score(pts, beta)
    for idx in range(pts.size):
        orig = pts.flat[idx]
        pts.flat[idx] = orig + eps
        fp = _soft_min_score(pts, beta)
        pts.flat[idx] = orig - eps
        fm = _soft_min_score(pts, beta)
        pts.flat[idx] = orig
        grad.flat[idx] = (fp - fm) / (2 * eps)
    return grad


def _local_polish(pts, max_iter=2000):
    """Deterministic local search: repeatedly lift the smallest triangle's points."""
    pts = pts.copy()
    cur = _score(pts)
    for _ in range(max_iter):
        if not _time_ok(2.0):
            break
        _, (i, j, k) = _smallest_triangle(pts)
        improved = False
        for pt in (i, j, k):
            for s in (0.003, 0.008, 0.015, 0.03, 0.06):
                if pt == i:
                    cand = _lift(pts, i, j, k, strength=s)
                elif pt == j:
                    cand = _lift(pts, j, i, k, strength=s)
                else:
                    cand = _lift(pts, k, i, j, strength=s)
                sc = _score(cand)
                if sc > cur + 1e-12:
                    pts, cur, improved = cand, sc, True
                    break
            if improved:
                break
        if not improved:
            break
    return pts, cur


def _generate_inits(rng):
    """Generate diverse initial configurations."""
    inits = []
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles13), np.sin(angles13)]))
    
    # Ring structures
    ring_structures = [
        [13], [1, 12], [1, 6, 6], [7, 6], [2, 11], [3, 10], [4, 9], [5, 8], [6, 7],
        [1, 3, 9], [1, 4, 8], [1, 5, 7], [2, 5, 6], [1, 2, 3, 7],
    ]
    for rc in ring_structures:
        nr = len(rc)
        base_r = [1.0 / (i + 1) for i in range(nr)]
        for ph in [0.0, np.pi / 6.0, np.pi / 3.0]:
            pts = []
            for i, count in enumerate(rc):
                if count == 0:
                    continue
                ang = ph + 2 * np.pi * np.arange(count) / count
                pts.append(base_r[i] * np.column_stack([np.cos(ang), np.sin(ang)]))
            if pts:
                inits.append(np.vstack(pts))
    
    # Random in unit disk
    for _ in range(8):
        r = np.sqrt(rng.rand(N))
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    
    # Jittered regular 13-gon
    for jit in [0.02, 0.05, 0.1]:
        rr = 1.0 + jit * rng.randn(N)
        inits.append(np.column_stack([rr * np.cos(angles13), rr * np.sin(angles13)]))
    
    return inits


def construct():
    rng = np.random.RandomState(42)
    
    inits = _generate_inits(rng)
    
    # Score all inits
    scored = []
    for ip in inits:
        if not _time_ok(30.0):
            break
        s = _score(ip)
        scored.append((s, ip))
    scored.sort(key=lambda x: -x[0])
    
    # Keep top distinct candidates
    top = []
    for s, p in scored:
        if not any(np.allclose(p, q, atol=1e-5) for q in top):
            top.append(p)
        if len(top) >= 10:
            break
    
    best_score = scored[0][0]
    best_pts = scored[0][1].copy()
    
    # Phase 1: Gradient-based optimization (L-BFGS-B) on top candidates
    for p in top[:6]:
        if not _time_ok(15.0):
            break
        try:
            # Flatten for optimizer
            x0 = p.flatten()
            
            def obj(x):
                pts = x.reshape(N, 2)
                return _soft_min_score(pts, beta=60.0)
            
            def grad(x):
                pts = x.reshape(N, 2)
                return _soft_min_grad(pts, beta=60.0).flatten()
            
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            cand_pts = res.x.reshape(N, 2)
            sc = _score(cand_pts)
            if sc > best_score:
                best_score, best_pts = sc, cand_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Local polish on the best and a few others
    candidates_to_polish = [best_pts]
    for p in top[:3]:
        if not np.allclose(p, best_pts, atol=1e-5):
            candidates_to_polish.append(p)
    
    for p in candidates_to_polish:
        if not _time_ok(10.0):
            break
        try:
            lp, ls = _local_polish(p, max_iter=3000)
            if ls > best_score:
                best_score, best_pts = ls, lp.copy()
        except Exception:
            pass
    
    # Phase 3: Another round of gradient optimization from the polished best
    if _time_ok(15.0):
        try:
            x0 = best_pts.flatten()
            
            def obj(x):
                pts = x.reshape(N, 2)
                return _soft_min_score(pts, beta=100.0)
            
            def grad(x):
                pts = x.reshape(N, 2)
                return _soft_min_grad(pts, beta=100.0).flatten()
            
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9})
            cand_pts = res.x.reshape(N, 2)
            sc = _score(cand_pts)
            if sc > best_score:
                best_score, best_pts = sc, cand_pts.copy()
        except Exception:
            pass
    
    # Final polish
    if _time_ok(5.0):
        try:
            lp, ls = _local_polish(best_pts, max_iter=4000)
            if ls > best_score:
                best_score, best_pts = ls, lp.copy()
        except Exception:
            pass
    
    return best_pts
