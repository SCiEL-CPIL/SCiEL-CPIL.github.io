"""Program for Erdos minimum overlap task using anti-periodic model + optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). A non-degenerate anti-periodic
solution therefore reaches near-0 (far below the seed value of 1).
We build a strong anti-periodic seed and refine it with a smooth objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    # triangular wave peaking at the middle of the first half
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    # center the first half around 0.5 so the anti-periodic completion
    # (0.5 - first half) stays within [0,1]
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    # fix tiny numerical drift
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    # free parameters: first half. second half = 1 - first (anti-periodic).
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 40})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def _polish_exact(p0, bounds, maxiter=500):
    """High-precision exact-objective polish with tight tolerances."""
    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value(unpack(p))

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-18,
                                "gtol": 1e-15, "maxls": 50})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return unpack(np.clip(p0, 0.0, 1.0))


def construct():
    t0 = time.time()

    h = make_seed()
    best = h.copy()
    best_val = value(best)

    # Progressive smoothing: start smooth, then sharpen to target the true max.
    for beta, iters in [(20.0, 200), (50.0, 200), (120.0, 200), (300.0, 300)]:
        if remaining() < 15.0:
            break
        h = refine(h, beta, iters)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Final exact-objective polish with multiple restarts to escape local minima.
    if remaining() > 20.0:
        p_best = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        # First polish from the current best
        h_polished = _polish_exact(p_best, bounds, maxiter=500)
        v = value(h_polished)
        if v < best_val:
            best, best_val = h_polished.copy(), v
            p_best = np.clip(best[:HALF], 0.0, 1.0).copy()

        # Multiple restarts with small perturbations to explore nearby basins
        rng = np.random.default_rng(42)
        n_restarts = 5
        for i in range(n_restarts):
            if remaining() < 15.0:
                break
            # Perturb with decreasing magnitude
            scale = 0.05 * (1.0 - i / n_restarts)
            p_perturbed = p_best + rng.normal(0, scale, size=HALF)
            p_perturbed = np.clip(p_perturbed, 0.0, 1.0)
            
            h_try = _polish_exact(p_perturbed, bounds, maxiter=300)
            v_try = value(h_try)
            if v_try < best_val:
                best, best_val = h_try.copy(), v_try
                p_best = np.clip(best[:HALF], 0.0, 1.0).copy()

        # One more high-precision polish from the final best
        if remaining() > 10.0:
            h_final = _polish_exact(p_best, bounds, maxiter=1000)
            v_final = value(h_final)
            if v_final < best_val:
                best, best_val = h_final.copy(), v_final

    return best
