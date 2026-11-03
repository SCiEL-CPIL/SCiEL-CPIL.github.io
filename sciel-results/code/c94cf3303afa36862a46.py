"""Heilbronn triangles: place 11 points in the equilateral triangle to maximize the
minimum triangle area.  Fresh design: symmetric seed + smooth epigraph polish."""
import time
import numpy as np

N = 11
H = np.sqrt(3.0) / 2.0
AREA = 0.5 * H            # area of the unit-side equilateral triangle
TIME_LIMIT = 115.0        # safety margin below the 150 s wall-clock kill
T_START = time.time()


def _time_left():
    return TIME_LIMIT - (time.time() - T_START)


def _all_areas(P):
    """Doubled area of every triple (P[i],P[j],P[k]), i<j<k, as a flat array."""
    n = P.shape[0]
    out = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = P[j] - P[i]
                v2 = P[k] - P[i]
                out[idx] = abs(v1[0] * v2[1] - v1[1] * v2[0])
                idx += 1
    return out


def _in_triangle(P):
    """True if every point is inside (or on) the unit-side equilateral triangle."""
    return bool(np.all((P[:, 0] >= -1e-9) &
                       (P[:, 1] >= -1e-9) &
                       (P[:, 0] + P[:, 1] / H <= 1.0 + 1e-9)))


def _project(P):
    """Project points into the triangle (vertex/edge clipping)."""
    Q = P.copy()
    Q[:, 0] = np.clip(Q[:, 0], 0.0, 1.0)
    Q[:, 1] = np.clip(Q[:, 1], 0.0, H)
    Q[:, 0] = np.minimum(Q[:, 0], 1.0 - Q[:, 1] / H)
    Q[:, 0] = np.clip(Q[:, 0], 0.0, 1.0)
    Q[:, 1] = np.clip(Q[:, 1], 0.0, H)
    return Q


def _initial_points():
    """3 corners + 3 edge midpoints + 3 interior ring points + 1 centre."""
    P = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, H],
        [0.5, 0.0], [0.75, H / 2.0], [0.25, H / 2.0],
        [1.0 / 3.0, H / 3.0], [2.0 / 3.0, H / 3.0], [0.5, 2.0 * H / 3.0],
        [0.5, H / 3.0],
    ])
    return P


def construct():
    P = _project(_initial_points())

    def objective(x):
        Q = x.reshape(N, 2)
        A = _all_areas(Q)
        # smooth epigraph: -logsumexp(A) + penalty for violating the triangle
        m = A.max()
        s = np.exp(A - m)
        logsumexp = m + np.log(s.sum())
        penalty = np.sum(np.maximum(0.0, Q[:, 0]) ** 2) \
                + np.sum(np.maximum(0.0, Q[:, 1]) ** 2) \
                + np.sum(np.maximum(0.0, Q[:, 0] + Q[:, 1] / H - 1.0) ** 2)
        return -logsumexp + 1e-3 * penalty

    best = P
    best_val = _all_areas(P).min() / AREA

    for alpha in (10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05):
        if _time_left() < 8.0:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(objective, P.ravel(), method="L-BFGS-B",
                           options={"maxiter": 2000, "maxls": 60, "ftol": 1e-15,
                                    "gtol": 1e-12, "maxfun": 6000})
            P = _project(res.x.reshape(N, 2))
        except Exception:
            P = _project(P)
        val = _all_areas(P).min() / AREA
        if val > best_val:
            best_val = val
            best = P.copy()

    return best


# EVOLVE-BLOCK-START
# (entry point construct() is defined above; kept here for block compatibility)
# EVOLVE-BLOCK-END
