"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
  * Start from the 12-gon + centre topology (interior point exploits the
    interior space; measured to beat the regular 13-gon).
  * Coarse global refinement with Powell and L-BFGS-B on a smooth
    soft-minimum triangle-area objective (avoids the kink of the exact min).
  * Fine local refinement that directly targets the current minimum-area
    triangle with a subgradient ascent step (raises the bottleneck triangle
    while keeping it non-degenerate), followed by a short Powell polish.
  * Everything is bounded by an explicit wall-clock deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 130.0  # safety margin well inside the 150 s wall-clock kill

# Precompute all C(N,3) index triples once (i<j<k).
def _make_triples(n):
    i, j, k = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                i.append(a); j.append(b); k.append(c)
    return np.array(i), np.array(j), np.array(k)

_TI, _TJ, _TK = _make_triples(N)
_NTRI = _TI.size  # 286


def _tri_areas(points):
    """Vectorized areas of all C(N,3) triangles (N=13 fixed)."""
    a = points[_TI]; b = points[_TJ]; c = points[_TK]
    v1 = b - a; v2 = c - a
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _hull_area(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull = _hull_area(points)
    if hull < 1e-10:
        return 0.0
    return float(np.min(_tri_areas(points))) / hull


def _min_tri(points):
    """Return (min_area, (i, j, k)) of the smallest triangle."""
    areas = _tri_areas(points)
    m = int(np.argmin(areas))
    return float(areas[m]), (int(_TI[m]), int(_TJ[m]), int(_TK[m]))


def _softmin_neg(x, beta):
    """Smooth objective: - (soft-min triangle area) / hull area."""
    pts = x.reshape(N, 2)
    areas = _tri_areas(pts)
    h = _hull_area(pts)
    if h < 1e-10:
        return 1e6
    soft = -np.logaddexp.reduce(np.log(np.maximum(areas, 1e-12)) / beta)
    return -soft / h


def _refine_coarse(pts, beta):
    """Powell then L-BFGS-B on the smooth objective."""
    best = pts.copy()
    best_s = _score(best)
    x0 = best.flatten()
    for method, kwargs in (
        ('Powell', dict(maxiter=400, ftol=1e-9, xtol=1e-9)),
        ('L-BFGS-B', dict(maxiter=300, ftol=1e-12)),
    ):
        try:
            res = minimize(lambda x: _softmin_neg(x, beta), x0, method=method, **kwargs)
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best = cand; best_s = s; x0 = res.x
        except Exception:
            pass
    return best, best_s


def _refine_fine(pts, budget):
    """Target the current minimum triangle with subgradient steps, then polish."""
    best = pts.copy()
    best_s = _score(best)
    lr = 0.02
    t0 = time.time()
    for _ in range(3000):
        if time.time() - t0 > budget:
            break
        a, (i, j, k) = _min_tri(best)
        if a < 1e-9:
            best = best + np.random.default_rng().normal(0, 0.01, (N, 2))
            continue
        P = best
        A, B, C = P[i], P[j], P[k]
        u = B - A; v = C - A
        cross = u[0] * v[1] - u[1] * v[0]
        s = 1.0 if cross >= 0 else -1.0
        g = np.zeros((N, 2))
        g[i] = -s * 0.5 * (v[1], -v[0])
        g[j] = s * 0.5 * (v[1], -v[0])
        g[k] = s * 0.5 * (-u[1], u[0])
        best = best + lr * g
        s2 = _score(best)
        if s2 > best_s:
            best_s = s2
        else:
            lr *= 0.8
            if lr < 1e-5:
                break
    # Short Powell polish on the exact (non-smooth) score.
    try:
        res = minimize(lambda x: -_score(x.reshape(N, 2)), best.flatten(),
                       method='Powell', maxiter=200, ftol=1e-10, xtol=1e-10)
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best = cand; best_s = s
    except Exception:
        pass
    return best, best_s


def _starts():
    """Topology-aware starting configurations (12-gon + centre favoured)."""
    starts = []
    a12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(a12), np.sin(a12)])
    starts.append(np.vstack([ring, [[0.0, 0.0]]]))
    for r in (0.3, 0.5, 0.7):
        ang = np.linspace(0, 2 * np.pi, 2, endpoint=False) + 0.4
        inner = r * np.column_stack([np.cos(ang), np.sin(ang)])
        starts.append(np.vstack([ring, inner]))
    a13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    starts.append(np.column_stack([np.cos(a13), np.sin(a13)]))
    return starts


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    best = None
    best_s = 0.0

    for idx, st in enumerate(_starts()):
        if time.time() > deadline - 18.0:
            break
        p, s = _refine_coarse(st, beta=0.02)
        if s > best_s:
            best_s, best = s, p.copy()
        if time.time() > deadline - 12.0:
            break
        p, s = _refine_fine(p, budget=8.0)
        if s > best_s:
            best_s, best = s, p.copy()

    if best is None:
        return _starts()[0]

    # Final polish on the overall best, if time remains.
    if time.time() < deadline - 6.0:
        p, s = _refine_fine(best, budget=6.0)
        if s > best_s:
            best_s, best = s, p

    return best
