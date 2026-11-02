"""Tuned Fourier-invariant Hermite-Gaussian test function.

Minimizes the upper bound B(a) = max_{lambda>=0} (1-g(lambda))/g(lambda)
over the three free Hermite coefficients (degrees 0, 4, 8). The degree-12
weight is fixed so that g(0) = 0.

This version uses an epigraph formulation with SLSQP to handle the
maximization constraint more robustly than simple gradient descent on
the max function.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
_DEADLINE = 210.0
_MARGIN = 10.0
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
    # a12 is fixed such that P(0) = 0
    # P(0) = a0*1 + a4*12 + a8*1680 + a12*798336 = 0
    # a12 = -(a0 + 12*a4 + 1680*a8) / 798336
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 798336.0
    h0, h4, h8, h12 = _hermite_values(lam)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * lam * lam)


def _bound_exact(a, grid=None):
    """Exact max of (1-g)/g over lambda>=0 on a dense grid with local refinement."""
    if grid is None:
        grid = np.linspace(0.0, 15.0, 10001)
    g = _g_of_lambda(a, grid)
    # Check for negative g
    if np.min(g) <= 1e-12:
        return 1e10 + (1e-12 - np.min(g)) * 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (1.0 - g) / g
    ratio = np.where(np.isfinite(ratio), ratio, 1e18)
    return float(np.max(ratio))


def _solve_epigraph(a_init, grid):
    """
    Solve min t s.t. g(lambda_i) >= 1/(1+t) for all i.
    Variables: [a0, a4, a8, t]
    """
    a0, a4, a8 = a_init
    # Initial t estimate
    g_init = _g_of_lambda(a_init, grid)
    min_g = np.min(g_init)
    if min_g <= 0:
        t_init = 10.0
    else:
        t_init = max(0.0, 1.0/min_g - 1.0)
    
    x_init = np.array([a0, a4, a8, t_init])
    
    def objective(x):
        a = x[:3]
        t = x[3]
        # Regularization to keep a bounded
        return t + 1e-6 * np.sum(a**2)
        
    def constraints(x):
        a = x[:3]
        t = x[3]
        # t must be positive
        c_t = t - 1e-6
        # g(lambda_i) >= 1/(1+t)
        g = _g_of_lambda(a, grid)
        target = 1.0 / (1.0 + t)
        # Constraint: g - target >= 0
        c_grid = g - target
        # Return as a single array or list of dicts
        # SLSQP expects a list of constraints or a single array if type='ineq'
        # Let's combine into one array
        return np.concatenate(([c_t], c_grid))
        
    # Define constraint type
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Bounds for a and t
    # a coefficients shouldn't be huge
    bounds_a = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    bounds_t = [(1e-6, 100.0)]
    bounds = bounds_a + bounds_t
    
    try:
        res = minimize(objective, x_init, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            return res.x[:3], float(res.fun - 1e-6 * np.sum(res.x[:3]**2)) # Remove reg from t
        else:
            return None, None
    except Exception as e:
        return None, None


def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    best_a = _SEED.copy()
    best_val = _bound_exact(_SEED)
    
    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - _START)
    
    # Grid for SLSQP
    # Use a moderate density for SLSQP to keep it fast
    grid_slsqp = np.linspace(0.0, 10.0, 200)
    
    # Phase 1: Try SLSQP from seed
    if remaining() > 10.0:
        a_opt, val_opt = _solve_epigraph(_SEED, grid_slsqp)
        if a_opt is not None:
            # Verify with exact bound
            v = _bound_exact(a_opt)
            if v < best_val:
                best_val = v
                best_a = a_opt.copy()
    
    # Phase 2: Multi-start SLSQP
    rng = np.random.default_rng(42)
    # Generate diverse starting points
    starts = [best_a.copy()]
    for _ in range(10):
        # Perturb seed
        scale = 0.1 * rng.random() + 0.01
        pert = rng.standard_normal(3) * scale
        starts.append(_SEED + pert)
        
        # Random points in reasonable range
        a0 = rng.uniform(-2.0, 0.0)
        a4 = rng.uniform(-0.2, 0.2)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))
        
    for s in starts:
        if remaining() < 5.0:
            break
        a_opt, val_opt = _solve_epigraph(s, grid_slsqp)
        if a_opt is not None:
            v = _bound_exact(a_opt)
            if v < best_val:
                best_val = v
                best_a = a_opt.copy()
                
    # Phase 3: Local polish with L-BFGS-B on exact bound (smoothed)
    if remaining() > 5.0:
        # Use a smoothed version of the exact bound for gradient
        def smooth_bound(a, temp=0.01):
            grid = np.linspace(0.0, 10.0, 500)
            g = _g_of_lambda(a, grid)
            if np.min(g) < 1e-4:
                return 1e10 + (1e-4 - np.min(g)) * 1e6
            ratio = (1.0 - g) / g
            m = np.max(ratio)
            s = np.exp((ratio - m) / temp)
            return m + temp * math.log(np.mean(s))
            
        try:
            res = minimize(smooth_bound, best_a, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                v = _bound_exact(res.x)
                if v < best_val:
                    best_val = v
                    best_a = res.x.copy()
        except Exception:
            pass
            
    return np.array(best_a, dtype=float)
# EVOLVE-BLOCK-END
