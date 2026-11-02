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

# Log-space transformation for better optimization of multi-scale coefficients
_LOG_SCALE = np.array([1.0, 0.1, 0.01])  # Scales for each coefficient
_LOG_OFFSET = np.array([0.0, 0.0, 0.0])  # Offsets in log space


def _to_log_space(a):
    """Convert from original space to log space for optimization."""
    # Handle sign: a0 is negative, a4 and a8 can be positive or negative
    # Use a transformation that preserves sign information
    sign0 = np.sign(a[0]) if a[0] != 0 else -1.0
    sign4 = np.sign(a[1]) if a[1] != 0 else 1.0
    sign8 = np.sign(a[2]) if a[2] != 0 else 1.0
    
    # For simplicity, use a smooth approximation
    # We'll optimize in original space but with better scaling
    return a


def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0,H_4,H_8,H_12 at points x."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    h0 = np.ones_like(x)
    h4 = 16.0 * x2**2 - 48.0 * x2 + 12.0
    h8 = 256.0 * x2**4 - 3584.0 * x2**3 + 13440.0 * x2**2 - 13440.0 * x2 + 1680.0
    h12 = (4096.0 * x2**6 - 212992.0 * x2**5 + 3317760.0 * x2**4
           - 25333440.0 * x2**3 + 100810080.0 * x2**2 - 201620160.0 * x2 + 798336.0)
    return h0, h4, h8, h12


def _g_of_lambda(a, lam):
    """g(lambda) = (a0 H0 + a4 H4 + a8 H8 + a12 H12)(lambda) * exp(-lambda^2/2)."""
    a0, a4, a8 = a
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 with adaptive local refinement."""
    if grid is None:
        # Coarse global grid
        grid = np.concatenate([
            np.linspace(0.0, 20.0, 100001),
            np.geomspace(1e-10, 40.0, 5000),
        ])
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(g > 1e-15, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.where(ratio > 0, ratio, 1e-18)
    
    idx = int(np.argmax(ratio))
    val = float(ratio[idx])
    x0 = float(grid[idx])
    
    # Adaptive local refinement: zoom in around the maximum
    for level in range(6):
        if remaining() < 1.0:
            break
        width = 0.1 * (0.5 ** level)
        lo = max(0.0, x0 - width)
        hi = x0 + width
        local = np.linspace(lo, hi, 5001)
        gl = _g_of_lambda(a, local)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl = (1.0 - gl) / gl
        rl = np.where(gl > 1e-15, rl, 1e18)
        rl = np.where(np.isfinite(rl), rl, 1e18)
        rl = np.where(rl > 0, rl, 1e-18)
        
        local_max = float(np.max(rl))
        local_idx = int(np.argmax(rl))
        x0 = float(local[local_idx])
        
        if local_max > val:
            val = local_max
    
    return val


def _bound_smooth(a, grid=None, temperature=0.01):
    """Smooth log-sum-exp surrogate for max (1-g)/g with improved stability."""
    if grid is None:
        # Use a non-uniform grid: dense near 0, sparse at large lambda
        grid = np.concatenate([
            np.linspace(0.0, 5.0, 20000),
            np.geomspace(5.0, 30.0, 10000)
        ])
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    # More careful handling of invalid values
    ratio = np.where(g > 1e-12, ratio, 1e18)
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    ratio = np.clip(ratio, 1e-10, 1e15)
    
    # Use a more stable log-sum-exp computation
    m = float(np.max(ratio))
    if m > 1e14:  # If the max is huge, just return it
        return m
    
    # Compute log-sum-exp stably
    shifted = ratio - m
    log_sum_exp = m + temperature * math.log(float(np.mean(np.exp(shifted / temperature))))
    return log_sum_exp


def remaining():
    if _START is None:
        return _DEADLINE - _MARGIN
    return _DEADLINE - _MARGIN - (time.time() - _START)


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    from scipy.optimize import minimize, differential_evolution

    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)

    # Phase 1: Differential evolution for global exploration with wider bounds
    if remaining() > 80.0:
        try:
            # Wider bounds to explore more of the parameter space
            bounds = [(-3.0, 0.0), (-0.2, 0.2), (-0.02, 0.02)]
            de_result = differential_evolution(
                _bound_smooth, bounds, args=(None, 0.03),
                seed=42, maxiter=80, popsize=25, tol=1e-10,
                polish=True, init='sobol',
                updating='deferred',
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

    # Phase 2: Multi-start L-BFGS-B on smooth surrogate with adaptive temperature
    rng = np.random.default_rng(123)
    starts = [best_a.copy()]
    
    # Random perturbations at multiple scales
    for scale in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        starts.append(best_a + scale * rng.standard_normal(3))
    
    # Structured starts based on physics - more comprehensive grid
    a0_vals = np.linspace(-1.3, -0.7, 15)
    a4_vals = np.linspace(-0.05, 0.08, 15)
    a8_vals = np.linspace(-0.001, 0.001, 10)
    
    # Sample a subset to keep computation manageable
    for i in range(0, len(a0_vals), 2):
        for j in range(0, len(a4_vals), 2):
            for k in range(0, len(a8_vals), 2):
                starts.append(np.array([a0_vals[i], a4_vals[j], a8_vals[k]]))
    
    # Additional structured starts near the seed
    for a0 in (-1.13, -1.1, -1.05, -1.0, -0.95):
        for a4 in (0.04, 0.03, 0.02, 0.01, 0.0):
            for a8 in (0.0003, 0.0002, 0.0001, 0.0, -0.0001):
                starts.append(np.array([a0, a4, a8]))
    
    # Use adaptive temperature: start high, decrease
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    for temp in temps:
        for s in starts:
            if remaining() < 5.0:
                break
            try:
                res = minimize(
                    lambda x, t=temp: _bound_smooth(x, temperature=t), s,
                    method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12, "maxls": 50},
                )
                cand = np.asarray(res.x, dtype=float)
                if np.all(np.isfinite(cand)):
                    v = _bound_exact(cand)
                    if v < best_val:
                        best_val = v
                        best_a = cand.copy()
            except Exception:
                continue

    # Phase 3: High-precision polish on exact objective with Nelder-Mead
    if remaining() > 10.0:
        try:
            # First try Nelder-Mead for robustness
            res = minimize(
                _bound_exact, best_a, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-14, "fatol": 1e-14, "adaptive": True},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 3b: L-BFGS-B polish on exact objective
    if remaining() > 5.0:
        try:
            res = minimize(
                _bound_exact, best_a, method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-18, "gtol": 1e-14, "maxls": 100},
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Phase 4: Fine grid polish with coordinate descent
    if remaining() > 3.0:
        for step in (0.001, 0.0005, 0.0002, 0.0001, 0.00005):
            if remaining() < 1.5:
                break
            improved = True
            while improved and remaining() > 0.8:
                improved = False
                for axis in range(3):
                    for direction in (-1, 1):
                        cand = best_a.copy()
                        cand[axis] += direction * step
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
                            improved = True

    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
