"""Heilbronn triangle problem: place 11 points to maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3.0)
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])


def _barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric coords to (N,2) cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2.0) * bary[:, 2]
    return np.column_stack([x, y])


def _min_triangle_area(points):
    """Compute the minimum area among all C(N,3) triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _negative_min_area(bary):
    """Objective: negative of min triangle area. Barycentric coords with simplex constraint."""
    # Normalize barycentric to sum to 1
    bary = bary / bary.sum(axis=1, keepdims=True)
    # Clip to [0, 1]
    bary = np.clip(bary, 0.0, 1.0)
    points = _barycentric_to_cartesian(bary)
    return -_min_triangle_area(points)


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    best_bary = None
    best_score = -float('inf')

    # Multiple random restarts
    rng = np.random.RandomState(42)
    max_restarts = 200

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random initial barycentric coordinates
        bary0 = rng.random((N, 3))
        bary0 = bary0 / bary0.sum(axis=1, keepdims=True)

        # Nelder-Mead optimization (doesn't need gradients, robust for non-smooth objectives)
        try:
            res = minimize(
                _negative_min_area,
                bary0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.success or res.fun < best_score:
                bary_opt = res.x
                bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
                bary_opt = np.clip(bary_opt, 0.0, 1.0)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_bary = bary_opt.copy()
        except Exception:
            continue

    if best_bary is None:
        # Fallback: random points
        bary = rng.random((N, 3))
        bary = bary / bary.sum(axis=1, keepdims=True)
        return _barycentric_to_cartesian(bary)

    return _barycentric_to_cartesian(best_bary)
