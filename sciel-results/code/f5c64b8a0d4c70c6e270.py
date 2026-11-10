"""Fresh design: Hilbert-curve seeding + Nelder-Mead on a smooth max-min surrogate,
with an explicit wall-clock deadline guard. Entry point: construct() -> (N,3) array."""

import time
import numpy as np
from scipy.optimize import minimize

N = 50
CUBE = 0.5            # half side length; centers must satisfy |c_ik| <= CUBE
DEADLINE = 120.0      # hard evaluator limit (seconds)
SAFETY = 2.0          # safety margin (seconds) before the hard limit


def _hilbert_3d(n, s=8):
    """Map integers 0..n-1 to well-spread points in [0,1)^3 via a 3-D Hilbert curve."""
    pts = []
    for i in range(n):
        x = y = z = 0
        t = i
        for _ in range(s * 3):
            rx = (t >> 1) & 1
            ry = t & 1
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            z ^= ry
            t >>= 2
        pts.append((x / s, y / s, z / s))
    return np.array(pts)


def _make_hilbert_initial(n):
    """Hilbert-spread initial centers inside [-CUBE, CUBE]^3 with tiny jitter."""
    base = _hilbert_3d(n)
    pts = (base - 0.5) * (2 * CUBE)
    rng = np.random.default_rng(12345)
    jitter = rng.uniform(-0.02, 0.02, size=pts.shape)
    pts = pts + jitter
    return np.clip(pts, -CUBE, CUBE)


def _score(pts):
    """Largest common radius (the task objective) for a set of centers."""
    p = np.asarray(pts)
    c = np.clip(p, -CUBE, CUBE)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    wall = (CUBE - np.abs(c)).min()
    return 0.5 * min(min_pair, 2.0 * wall)


def _surrogate(pts):
    """Smooth differentiable surrogate: -logsumexp(-d/tau) ~= max-min distance."""
    p = np.asarray(pts)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d = d[d < np.inf]
    tau = 0.05
    x = -d / tau
    x = x - x.max()
    return -tau * np.log(np.exp(x).sum() + 1e-12)


def _nelder_mead_optimize(pts, tau, time_limit):
    """Run Nelder-Mead on the surrogate, stopping if the wall-clock limit is hit."""
    t0 = time.time()
    x0 = np.asarray(pts, dtype=float).ravel()
    x0 = np.clip(x0, -CUBE, CUBE)
    best = x0.copy()
    best_val = _surrogate(best.reshape(-1, 3))

    def obj(x):
        nonlocal best, best_val
        if time.time() > time_limit:
            return 0.0
        xx = np.clip(x, -CUBE, CUBE).reshape(-1, 3)
        v = _surrogate(xx)
        if v > best_val:
            best_val = v
            best = xx.ravel().copy()
        return -v

    res = minimize(
        obj, x0, method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True},
    )
    if res.x is not None and np.all(np.isfinite(res.x)):
        cand = np.clip(res.x, -CUBE, CUBE).reshape(-1, 3)
        if _surrogate(cand) > best_val:
            best = cand.ravel().copy()
    return best.reshape(-1, 3)


def construct():
    t_start = time.time()
    deadline = t_start + (DEADLINE - SAFETY)

    def remaining():
        return max(0.0, deadline - time.time())

    # --- Phase 0: Hilbert-spread seed (valid fallback) ---
    best = _make_hilbert_initial(N)
    best_score = _score(best)

    # --- Phase 1: Nelder-Mead on a coarse surrogate (cheap, global-ish) ---
    if remaining() > 1.0:
        x = _nelder_mead_optimize(best, tau=0.05, time_limit=deadline)
        if _score(x) > best_score:
            best, best_score = x, _score(x)

    # --- Phase 2: refine with a tighter surrogate on the best so far ---
    if remaining() > 1.0:
        x = _nelder_mead_optimize(best, tau=0.02, time_limit=deadline)
        if _score(x) > best_score:
            best, best_score = x, _score(x)

    # --- Phase 3: one final polish if time allows ---
    if remaining() > 1.0:
        x = _nelder_mead_optimize(best, tau=0.01, time_limit=deadline)
        if _score(x) > best_score:
            best, best_score = x, _score(x)

    # --- Final validity clamp (never emit out-of-cube / non-finite / coincident) ---
    best = np.clip(best, -CUBE, CUBE)
    best = np.nan_to_num(best, nan=0.0, posinf=CUBE, neginf=-CUBE)
    return best
