"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound B(a) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over the three free Hermite coefficients (degrees 0, 4, 8). The degree-12
weight is fixed so that g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 25.0
_START = None


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    h8 = 256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4 - 13440.0 * x**2 + 1680.0
    h12 = (4096.0 * x**12 - 212992.0 * x**10 + 3317760.0 * x**8
           - 25333440.0 * x**6 + 100810080.0 * x**4 - 201620160.0 * x**2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with local refinement."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.0, 25.0, 150001),
            np.geomspace(1e-8, 50.0, 5000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    if 0.0 < x0 < 50.0:
        local = np.linspace(max(0.0, x0 - 0.02), x0 + 0.02, 2001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        val = max(val, float(np.max(rl)))
    return val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.0, 25.0, 50001)
    g = _g_of_lambda(a, grid)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-10, ratio, 1e15)
    ratio = np.where(np.isfinite(ratio), ratio, 1e15)
    ratio = np.clip(ratio, -1e6, 1e15)
    
    m = float(np.max(ratio))
    temp = max(temperature, 1e-6)
    s = np.exp((ratio - m) / temp)
    return m + temp * math.log(float(np.mean(s)))


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    from scipy.optimize import minimize, differential_evolution

    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    # Phase 1: Differential evolution for global exploration
    if remaining() > 60.0:
        try:
            bounds = [(-5.0, 0.0), (-0.5, 0.5), (-0.05, 0.05)]
            de_result = differential_evolution(
                _bound_smooth, bounds, args=(None, 0.02),
                seed=42, maxiter=80, popsize=25, tol=1e-8,
                polish=True, init='sobol',
            )
            if de_result.success or de_result.fun < best_val:
                cand = np.asarray(de_result.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
        except Exception:
            pass

    # Phase 2: Multi-start L-BFGS-B on smooth surrogate
    rng = np.random.default_rng(123)
    starts = [best_a.copy()]
    for scale in (0.01, 0.05, 0.1, 0.3, 0.8):
        starts.append(best_a + scale * rng.standard_normal(3))
    
    # Structured starts focused on likely region
    for a0 in np.linspace(-2.0, -0.5, 7):
        for a4 in np.linspace(-0.05, 0.05, 5):
            for a8 in np.linspace(-0.003, 0.003, 3):
                starts.append(np.array([a0, a4, a8]))

    temps = [0.05, 0.02, 0.01, 0.005]
    for temp in temps:
        for s in starts:
            if remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue

    # Phase 3: Nelder-Mead on smooth surrogate (no gradient needed, robust)
    if remaining() > 5.0:
        try:
            res = minimize(
                lambda x: _bound_smooth(x, temperature=0.005), best_a,
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10,
                         "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 4: Final L-BFGS-B polish with very low temperature
    if remaining() > 3.0:
        try:
            res = minimize(
                lambda x: _bound_smooth(x, temperature=0.001), best_a,
                method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-13},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 5: Fine grid polish around the best point
    if remaining() > 2.0:
        for step in (0.001, 0.0005, 0.0002):
            if remaining() < 1.0:
                break
            for i in (-2, -1, 0, 1, 2):
                for j in (-2, -1, 0, 1, 2):
                    for k in (-2, -1, 0, 1, 2):
                        if remaining() < 0.5:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
