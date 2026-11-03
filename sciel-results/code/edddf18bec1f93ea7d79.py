"""Optimise 50 equal circles inside the unit circle: maximise the common radius.

Score = min( min_{i<j} |c_i - c_j| / 2 ,  min_i (1 - |c_i|) ).
Strategy (optimisation family, multi-ring, smooth):
  * deterministic multi-ring initial guess (already high-scoring),
  * smooth soft-min (log-sum-exp) objective maximised with L-BFGS-B,
  * deterministic multi-restart (no RNG -> reproducible),
  * final high-precision polish pass,
  * explicit wall-clock deadline guard with a safety margin.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
T_LIMIT = 120.0      # evaluator hard limit (s)
T_SAFE = 105.0       # safety margin: stop searching, return best (s)
T_START = time.time()


def _score(points):
    """Exact task score in double precision (or -inf if invalid)."""
    if points is None or points.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    d = 1.0 - np.linalg.norm(points, axis=1)
    if np.any(d < -1e-12):
        return -np.inf
    p = points
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, 1)
    mn = np.min(dist[iu]) / 2.0
    return float(min(mn, np.min(d)))


def _softmin_obj(pts, K):
    """Smooth surrogate: -logsumexp(-K * (2x - 1))  ~  -softmin(2x - 1).

    Minimising this maximises the soft-min of the scaled constraints
    (2r_pair - 1, 2r_margin - 1), i.e. maximises the minimum radius.
    """
    x = pts.reshape(-1)
    d = 1.0 - np.linalg.norm(pts, axis=1)
    p = pts
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, 1)
    v = np.concatenate([dist[iu] - 1.0, 2.0 * d - 1.0])
    return float(logsumexp(-K * v) / K)


def _ring_guess(rings):
    """Deterministic multi-ring guess: concentric rings at hexagonal spacings.

    rings = list of (radius, count).  The centre (0,0) is always included first.
    """
    pts = [[0.0, 0.0]]
    for rad, cnt in rings:
        if cnt <= 0:
            continue
        a = np.arange(cnt) * 2.0 * np.pi / cnt
        pts.extend(np.c_[rad * np.cos(a), rad * np.sin(a)].tolist())
    return np.array(pts[:N])


def _polish(x, K, tol, maxiter):
    """L-BFGS-B minimisation of the soft-min objective from x."""
    try:
        res = minimize(_softmin_obj, x, args=(K,), method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": tol,
                                "gtol": 1e-10, "maxls": 30})
        return res.x.reshape(N, 2), float(res.fun)
    except Exception:
        return x.reshape(N, 2), np.inf


def construct():
    global T_START
    T_START = time.time()

    # --- deterministic multi-ring initial guesses (centre + concentric rings) ---
    guesses = [
        _ring_guess([(0.30, 12), (0.60, 12), (0.87, 25)]),
        _ring_guess([(0.34, 10), (0.65, 12), (0.87, 27)]),
        _ring_guess([(0.28, 10), (0.56, 12), (0.80, 14), (0.93, 13)]),
        _ring_guess([(0.32, 11), (0.62, 11), (0.90, 27)]),
        _ring_guess([(0.36, 12), (0.70, 13), (0.90, 24)]),
        _ring_guess([(0.30, 9), (0.60, 11), (0.85, 13), (0.95, 16)]),
    ]

    best_x, best_val = None, -np.inf
    for g in guesses:
        v = _score(g)
        if v > best_val:
            best_val, best_x = v, g.copy()

    # --- smooth optimisation: deterministic multi-restart, coarse then fine ---
    K_coarse = 150.0
    K_fine = 400.0
    starts = [best_x] + guesses
    for g in starts:
        if time.time() - T_START > T_SAFE:
            break
        x, _ = _polish(g, K_coarse, tol=1e-8, maxiter=350)
        if time.time() - T_START > T_SAFE:
            break
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()

    # --- final high-precision polish pass (tighter tolerance, more iterations) ---
    for x0 in [best_x]:
        if time.time() - T_START > T_SAFE:
            break
        x, _ = _polish(x0, K_fine, tol=1e-11, maxiter=1200)
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()

    # --- final safety polish if time remains (bounded, cheap) ---
    if best_x is not None and time.time() - T_START < T_SAFE - 3.0:
        x, _ = _polish(best_x, K_fine, tol=1e-12, maxiter=600)
        v = _score(x)
        if v > best_val:
            best_val, best_x = v, x.copy()

    # --- validate; fall back to the best valid ring guess if optimisation broke ---
    out = best_x
    if out is None or out.shape != (N, 2) or not np.all(np.isfinite(out)) \
            or _score(out) < -1e-9:
        for g in guesses:
            if _score(g) > -1e-9:
                return g
        # absolute last resort: a safe, valid single ring
        return _ring_guess([(0.87, 49)])
    return out
