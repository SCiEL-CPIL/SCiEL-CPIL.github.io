"""QAOA angle design: exact combinatorial evaluation + per-graph local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 150.0
_MARGIN = 12.0          # safety margin before the hard kill
_WARM = [
    (0.40, 0.30), (0.25, 0.20), (0.55, 0.15),
    (0.30, 0.45), (0.15, 0.35), (0.65, 0.25),
    (0.45, 0.45), (0.20, 0.10),
]


def _time_left():
    return _DEADLINE - _MARGIN - time.time()


def _precompute(n, edges):
    m = 1 << n
    bits = np.arange(m, dtype=np.int64)
    c = np.zeros(m, dtype=np.float64)
    for a, b in edges:
        ca = (bits >> a) & 1
        cb = (bits >> b) & 1
        c += (ca ^ cb)
    idx = np.arange(n, dtype=np.int64)
    u1 = (bits >> idx) & 1
    u1 = 1.0 - 2.0 * u1.astype(np.float64)
    u2 = 1.0 - 2.0 * (idx[None, :] & bits[:, None]).astype(np.float64)
    return c, u1, u2


def _expected_cut(c, u1, u2, g1, g2, b1, b2):
    cg1 = c * np.cos(g1)
    cg2 = c * np.cos(g2)
    cg12 = c * np.cos(g1 - g2)
    v1 = np.cos(b1) + np.sin(b1) * u1
    v2 = np.cos(b2) + np.sin(b2) * u2
    t1 = cg1 @ v1
    t2 = cg2 @ v2
    t3 = cg12 @ (u1 * v2)
    return (t1 + t2 - 2.0 * t3) * 0.5


def _exact_optimum(c):
    return float(c.max())


def _neg_ratio(params, c, u1, u2, opt):
    g1, b1, g2, b2 = params
    ec = _expected_cut(c, u1, u2, g1, g2, b1, b2)
    return -ec / opt


def _optimize(c, u1, u2, opt, p):
    best = None
    best_val = -np.inf
    for g1, b1 in _WARM:
        if _time_left() <= 0.05:
            break
        x0 = np.array([g1, b1, g1, b1], dtype=np.float64)
        try:
            res = minimize(_neg_ratio, x0, args=(c, u1, u2, opt), method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-9, "fatol": 1e-12,
                                    "adaptive": True})
            val = -res.fun
        except Exception:
            try:
                res = minimize(_neg_ratio, x0, args=(c, u1, u2, opt), method="L-BFGS-B",
                               options={"maxiter": 200, "gtol": 1e-10})
                val = -res.fun
            except Exception:
                val = -_neg_ratio(x0, c, u1, u2, opt)
        if val > best_val:
            best_val = val
            best = res.x if hasattr(res, "x") else x0
    return best, best_val


def angles(n, edges, p):
    c, u1, u2 = _precompute(n, edges)
    opt = _exact_optimum(c)
    if opt <= 0:
        return np.zeros((p, 2), dtype=np.float64)

    best, _ = _optimize(c, u1, u2, opt, p)
    if best is None:
        best = np.array([0.35, 0.25, 0.35, 0.25], dtype=np.float64)

    g1, b1, g2, b2 = best
    out = np.empty((p, 2), dtype=np.float64)
    for i in range(p):
        out[i, 0] = g1 if i % 2 == 0 else g2
        out[i, 1] = b1 if i % 2 == 0 else b2
    return out
# EVOLVE-BLOCK-END
