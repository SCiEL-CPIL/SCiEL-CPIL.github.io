"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound B(a) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over the three free Hermite coefficients (degrees 0, 4, 8). The degree-12
weight is fixed so that g(0) = 0.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Seed coefficients from the current program
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_START_TIME = None
_DEADLINE = 170.0  # Safety margin from 180s limit

def _hermite_values(x):
    """Vectorised physicists' Hermite polynomials H_0, H_4, H_8, H_12 at points x."""
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
    # a12 is fixed to ensure g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=798336
    # a0*1 + a4*12 + a8*1680 + a12*798336 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)

def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid."""
    if grid is None:
        # Focus on region where g is significant and positive
        # g decays rapidly for large lambda. 
        # The ratio (1-g)/g is large when g is small (near 0 or negative) or when g is close to 1.
        # We need to scan a wide range to ensure we catch the maximum.
        grid = np.linspace(0.0, 30.0, 20000)
    
    g = _g_of_lambda(a, grid)
    
    # If g <= 0, the ratio is effectively infinite (or very large)
    # We want to avoid this.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    
    # Mask invalid values (g <= 0 or inf)
    valid = (g > 1e-12) & np.isfinite(ratio)
    if not np.any(valid):
        return 1e18
        
    ratio_valid = ratio[valid]
    max_val = np.max(ratio_valid)
    
    # Refine around the maximum
    idx = np.argmax(ratio)
    if valid[idx]:
        x0 = grid[idx]
        # Local refinement
        local_grid = np.linspace(max(0.0, x0 - 0.5), x0 + 0.5, 1000)
        g_local = _g_of_lambda(a, local_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_local = (1.0 - g_local) / g_local
        valid_local = (g_local > 1e-12) & np.isfinite(ratio_local)
        if np.any(valid_local):
            max_val = max(max_val, np.max(ratio_local[valid_local]))
            
    return float(max_val)

def _bound_smooth(a, grid=None, temperature=0.1):
    """Smooth log-sum-exp surrogate for max (1-g)/g."""
    if grid is None:
        # Coarser grid for speed during optimization
        grid = np.linspace(0.0, 20.0, 5000)
    
    g = _g_of_lambda(a, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    
    # Handle invalid values: if g <= 0, ratio is bad. 
    # We clip ratio to a large number to penalize, but keep it finite for log-sum-exp.
    # Actually, if g <= 0, we want the objective to be very high.
    # Let's set ratio to a large positive number where g is small/negative.
    ratio = np.where(g > 1e-12, ratio, 1e10)
    ratio = np.where(np.isfinite(ratio), ratio, 1e10)
    
    # Clip to prevent overflow in exp
    ratio = np.clip(ratio, 1e-12, 1e10)
    
    m = np.max(ratio)
    # Soft max approximation: m + T * log(mean(exp((x-m)/T)))
    # To be stable, we use: m + T * log( sum(exp((x-m)/T)) / N )
    # This is equivalent to log-sum-exp scaled by T.
    s = np.exp((ratio - m) / temperature)
    val = m + temperature * math.log(np.mean(s))
    
    return float(val)

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    from scipy.optimize import minimize, differential_evolution

    best_a = _SEED.copy()
    # Calculate the seed bound to know our baseline
    seed_val = _bound_exact(_SEED)
    best_val = seed_val
    
    def remaining():
        return _DEADLINE - (time.time() - _START_TIME)

    # Phase 1: Global Optimization using Differential Evolution
    # We use a smooth surrogate with a moderate temperature to guide DE.
    if remaining() > 60.0:
        try:
            # Bounds: a0 is likely negative, a4, a8 small
            # Based on seed: a0=-1.13, a4=0.04, a8=0.0003
            bounds = [(-2.0, 0.0), (-0.2, 0.2), (-0.01, 0.01)]
            
            # DE can be slow. Use a small population and few iterations.
            # We rely on the smooth surrogate to be fast.
            de_result = differential_evolution(
                _bound_smooth, 
                bounds, 
                args=(None, 0.5), # Temperature 0.5 for smoother landscape
                seed=42, 
                maxiter=30, 
                popsize=10, 
                tol=1e-6,
                polish=True,
                init='sobol'
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

    # Phase 2: Local Refinement using L-BFGS-B
    # Use decreasing temperatures to sharpen the surrogate.
    if remaining() > 20.0:
        # Start from the best candidate found
        starts = [best_a.copy()]
        
        # Add some random perturbations to escape local minima if stuck
        rng = np.random.default_rng(123)
        for scale in [0.01, 0.05, 0.1]:
            starts.append(best_a + scale * rng.standard_normal(3))
        
        for temp in [0.2, 0.05, 0.01, 0.001]:
            if remaining() < 5.0:
                break
            for s in starts:
                if remaining() < 2.0:
                    break
                try:
                    res = minimize(
                        lambda x: _bound_smooth(x, temperature=temp), 
                        s, 
                        method="L-BFGS-B",
                        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9}
                    )
                    cand = np.asarray(res.x, dtype=float)
                    if np.all(np.isfinite(cand)):
                        v = _bound_exact(cand)
                        if v < best_val:
                            best_val = v
                            best_a = cand.copy()
                            # Update starts with the new best for subsequent temps
                            starts[0] = best_a.copy()
                except Exception:
                    continue

    # Phase 3: Final Polish on Exact Objective
    # The exact objective is non-smooth (max), but L-BFGS-B can often handle it 
    # if we are very close to the optimum, treating it as a black box.
    if remaining() > 5.0:
        try:
            res = minimize(
                _bound_exact, 
                best_a, 
                method="L-BFGS-B",
                options={"maxiter": 50, "ftol": 1e-10, "gtol": 1e-8}
            )
            cand = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(cand)):
                v = _bound_exact(cand)
                if v < best_val:
                    best_val = v
                    best_a = cand.copy()
        except Exception:
            pass

    # Ensure we return a valid array
    return np.array(best_a, dtype=float)

# EVOLVE-BLOCK-END
