"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the classical upper bound on A(f)*A(Fourier(f)) over the three free
Hermite coefficients (degrees 0, 4, 8).  The degree-12 weight is fixed so that
g(0) = 0.  We minimize the bound
    B(a) = max_{lambda>=0} (1 - g(lambda)) / g(lambda)
using a smooth log-sum-exp surrogate for the outer max during optimization,
then a high-precision polish on the exact (max) objective.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 20.0          # safety margin (seconds) before the hard kill
_START = None


def _h2n(n):
    """H_0(0), H_4(0), H_8(0), H_12(0) for the physicists' Hermite polynomials."""
    return 1.0, 12.0, 1680.0, 798336.0


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    h0 = np.ones_like(x)
    h4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    h8 = (256.0 * x**8 - 3584.0 * x**6 + 13440.0 * x**4
          - 13440.0 * x**2 + 1680.0)
    h12 = (4096.0 * x**12 - 212992.0 * x**10 + 3317760.0 * x**8
           - 25333440.0 * x**6 + 100810080.0 * x**4
           - 201620160.0 * x**2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2),
    with a12 chosen so g(0)=0."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 with adaptive refinement."""
    if grid is None:
        # Multi-scale grid: fine near origin, coarser far out
        grid = np.concatenate([
            np.linspace(0.0, 5.0, 50001),
            np.linspace(5.0, 15.0, 30000),
            np.linspace(15.0, 40.0, 10000),
            np.geomspace(1e-8, 0.01, 1000),
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-14, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    # Adaptive refinement: 3 levels
    for level in range(3):
        half = 0.02 / (level + 1)
        lo = max(0.0, x0 - half)
        hi = x0 + half
        local = np.linspace(lo, hi, 2001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-14, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        idx2 = int(np.argmax(rl))
        val = max(val, float(rl[idx2]))
        x0 = float(local[idx2])
    return val


def _bound_smooth(a, grid=None, temperature=0.02):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        grid = np.linspace(0.0, 30.0, 60001)
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-12, 1e15)
    m = float(np.max(ratio))
    s = np.exp((ratio - m) / temperature)
    softmax = m + temperature * math.log(float(np.mean(s)))
    return softmax


def _bound_smooth_analytic(a, temperature=0.02):
    """Smooth log-sum-exp surrogate with log-space computation for stability."""
    grid = np.linspace(0.0, 25.0, 15001)
    g = _g_of_lambda(a, grid)
    # Work in log-space: log((1-g)/g) = log(1-g) - log(g)
    # For g near 0: ratio is huge; for g > 1: ratio is negative
    with np.errstate(divide="ignore", invalid="ignore"):
        log_g = np.log(np.clip(g, 1e-15, None))
        log_1mg = np.log(np.clip(1.0 - g, 1e-15, None))
        log_ratio = log_1mg - log_g
    # Mask invalid entries
    log_ratio = np.where(np.isfinite(log_ratio), log_ratio, 20.0)  # log(1e18) ≈ 41
    log_ratio = np.clip(log_ratio, -20.0, 45.0)
    # Log-sum-exp in log-space
    m = float(np.max(log_ratio))
    s = np.exp(log_ratio - m)
    result = m + math.log(float(np.mean(s)))
    return math.exp(result) if result < 700 else 1e18


def construct():
    global _START
    if _START is None:
        _START = time.time()
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)

    from scipy.optimize import minimize, differential_evolution

    # --- Phase 1: Global search with differential evolution on smooth surrogate ---
    if remaining() > 40.0:
        try:
            bounds = [(-2.5, 0.0), (-0.15, 0.15), (-0.002, 0.002)]
            result_de = differential_evolution(
                lambda x: _bound_smooth_analytic(x, temperature=0.08),
                bounds=bounds,
                seed=42,
                maxiter=150,
                tol=1e-7,
                popsize=20,
                polish=True,
                init="sobol",
            )
            cand = np.asarray(result_de.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Second DE run with different seed if time permits
    if remaining() > 30.0:
        try:
            bounds = [(-2.5, 0.0), (-0.15, 0.15), (-0.002, 0.002)]
            result_de2 = differential_evolution(
                lambda x: _bound_smooth_analytic(x, temperature=0.05),
                bounds=bounds,
                seed=123,
                maxiter=120,
                tol=1e-7,
                popsize=15,
                polish=True,
            )
            cand = np.asarray(result_de2.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # --- Phase 2: Multi-start local optimization on smooth surrogate ---
    rng = np.random.default_rng(0)
    starts = [best_a.copy(), _SEED.copy()]
    # Structured starts with varied scales
    for scale in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15):
        starts.append(best_a + scale * rng.standard_normal(3))
        starts.append(_SEED + scale * rng.standard_normal(3))
    # Axis-aligned perturbations
    for i in range(3):
        for sign in (-1, 1):
            for scale in (0.01, 0.03, 0.06):
                p = best_a.copy()
                p[i] += sign * scale
                starts.append(p)
    # Hand-picked diverse starts
    hand_picks = [
        [-1.1, 0.04, 0.0002],
        [-1.05, 0.035, 0.0003],
        [-1.15, 0.045, 0.00015],
        [-1.08, 0.042, 0.00025],
        [-1.12, 0.038, 0.00018],
        [-1.2, 0.05, 0.0001],
        [-1.0, 0.03, 0.0004],
        [-1.07, 0.037, 0.00022],
        [-1.13, 0.041, 0.00017],
        [-1.18, 0.048, 0.00012],
    ]
    for hp in hand_picks:
        starts.append(np.array(hp, dtype=float))

    # Run L-BFGS-B from multiple starts with decreasing temperature
    for temp in (0.08, 0.04, 0.02, 0.01):
        for s in starts:
            if remaining() < 4.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth_analytic(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
                )
            except Exception:
                continue
            cand = np.asarray(res.x, dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            v = _bound_exact(cand)
            if v < best_val:
                best_val = v
                best_a = cand.copy()

    # --- Phase 3: High-precision polish on the exact (max) objective ---
    if remaining() > 15.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 800, "ftol": 1e-15, "gtol": 1e-12},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Second polish from a slightly perturbed point
    if remaining() > 10.0:
        try:
            pert = best_a + 0.002 * np.array([1.0, -1.0, 0.5])
            res2 = minimize(
                _bound_exact, pert, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
            )
            cand = np.asarray(res2.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # --- Phase 4: Fine local search around the best point ---
    if remaining() > 5.0:
        for step in (0.005, 0.002, 0.001, 0.0005, 0.0002):
            if remaining() < 2.0:
                break
            # 27-point neighborhood
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        if remaining() < 1.0:
                            break
                        cand = best_a + step * np.array([i, j, k], dtype=float)
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()

    # --- Phase 5: Final Nelder-Mead polish ---
    if remaining() > 3.0:
        try:
            res_nm = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-14},
            )
            cand = np.asarray(res_nm.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
