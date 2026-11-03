"""Heilbronn convex regions n=14 — centrally-symmetric parameterization + multi-start local search."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 14
HALF = N // 2  # 7 independent points; full set = {p_i, -p_i}

DEADLINE = 110.0  # safety margin below 120s


def _all_points(params):
    """params: shape (7,2) -> full 14 points (7 + their negatives)."""
    return np.vstack([params, -params])


def _min_triangle_area(points):
    """Minimum triangle area over all triples (vectorized-ish)."""
    n = points.shape[0]
    min_a = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = 0.5 * abs(np.cross(points[j] - points[i], points[k] - points[i]))
                if a < min_a:
                    min_a = a
    return min_a


def _hull_area(points):
    """Convex hull area via monotonic chain (no scipy dependency for speed)."""
    pts = np.unique(np.round(points, 12), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(params):
    """Objective to maximize: min_triangle_area / hull_area. Return negative for minimization."""
    pts = _all_points(params)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e9  # bad
    mta = _min_triangle_area(pts)
    return -(mta / ha)


def _neg_objective(params):
    return _score(params)


def construct():
    t0 = time.time()
    best_params = None
    best_val = -np.inf

    rng = np.random.default_rng(2024)

    # Fix a scale: constrain the 7 independent points to lie within a unit disk-ish region.
    # We sample in a disk of radius 1 to keep the hull area O(1).

    def sample_initial():
        r = rng.uniform(0.0, 1.0, HALF)
        th = rng.uniform(0.0, 2 * np.pi, HALF)
        p = np.column_stack([r * np.cos(th), r * np.sin(th)])
        # avoid exact center duplicates
        return p

    # A few structured starts to seed good basins
    structured = []
    # Regular 7-gon (centrally symmetric pair set)
    th = np.linspace(0, 2 * np.pi, HALF, endpoint=False)
    structured.append(np.column_stack([np.cos(th), np.sin(th)]))
    # Two concentric rings
    for rad in (0.5, 0.85, 1.0):
        th = np.linspace(0, 2 * np.pi, HALF, endpoint=False) + rng.uniform(0, 0.5)
        structured.append(np.column_stack([rad * np.cos(th), rad * np.sin(th)]))

    def try_optimize(x0):
        nonlocal best_params, best_val
        # Nelder-Mead on the 14-dim flattened params
        x0f = x0.ravel()
        try:
            res = minimize(_neg_objective, x0f, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10,
                                    "adaptive": True})
            val = -res.fun
            if val > best_val and np.isfinite(val):
                best_val = val
                best_params = res.x.reshape(HALF, 2)
        except Exception:
            pass
        # also evaluate the raw start
        v = -_score(x0)
        if v > best_val and np.isfinite(v):
            best_val = v
            best_params = x0.copy()

    # Structured starts first
    for s in structured:
        if time.time() - t0 > DEADLINE:
            break
        try_optimize(s)

    # Random restarts until deadline
    n_random = 0
    while time.time() - t0 < DEADLINE - 5.0:
        x0 = sample_initial()
        try_optimize(x0)
        n_random += 1
        # adaptive: if we've been running a while, reduce iterations implicitly via deadline

    if best_params is None:
        # fallback: regular 7-gon
        th = np.linspace(0, 2 * np.pi, HALF, endpoint=False)
        best_params = np.column_stack([np.cos(th), np.sin(th)])

    # Final refinement with a tighter Nelder-Mead if time allows
    if time.time() - t0 < DEADLINE - 10.0:
        try:
            res = minimize(_neg_objective, best_params.ravel(), method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12,
                                    "adaptive": True})
            v = -res.fun
            if v > best_val and np.isfinite(v):
                best_val = v
                best_params = res.x.reshape(HALF, 2)
        except Exception:
            pass

    return _all_points(best_params)
