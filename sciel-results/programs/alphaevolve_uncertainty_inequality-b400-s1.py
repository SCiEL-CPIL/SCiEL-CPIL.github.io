"""Rewritten program: refined optimization for Hermite-Gaussian uncertainty bound."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time
import math


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties.
    Since all degrees are multiples of 4, F(k) = sqrt(2*pi)*g(k),
    so the product is sqrt(2*pi) * (sup|x*g(x)|)^2.
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 47900160.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Dense grid for accurate supremum
    N = 2000
    x = np.linspace(-9, 9, N)
    x2 = x ** 2
    
    H0 = np.ones_like(x)
    H4 = 16*x2*x2 - 48*x2 + 12
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    H8 = 256*x8 - 4480*x6 + 26880*x4 - 6720*x2 + 1680
    H12 = 4096*x12 - 245760*x10 + 5529600*x8 - 45158400*x6 + 164430720*x4 - 222772500*x2 + 47900160
    
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x2 / 2.0)
    xg = x * g_vals
    Af = np.max(np.abs(xg))
    
    bound = math.sqrt(2 * math.pi) * Af * Af
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    try:
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    except:
        return 1e10


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(-3.0, 0.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=15,
                tol=1e-10, seed=42,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Local refinement with multiple starting points
    start_points = [
        best_x.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.15, 0.035, 0.00028]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-1.0, 0.05, 0.0004]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, sp, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
