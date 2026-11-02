"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.integrate import quad
from scipy.special import factorial

_DEADLINE = time.time() + 195

# Precompute H_n(0) values
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 3024.0
_H12_0 = 10321920.0

def _poly_coeffs(a0, a4, a8):
    """Return coefficients of P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 as list [c0,...,c12]."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    coeffs = np.zeros(13)
    coeffs[0] += a0
    coeffs[0] += a4 * 12.0
    coeffs[2] += a4 * (-48.0)
    coeffs[4] += a4 * 16.0
    coeffs[0] += a8 * 10080.0
    coeffs[2] += a8 * (-34560.0)
    coeffs[4] += a8 * 23040.0
    coeffs[6] += a8 * (-4608.0)
    coeffs[8] += a8 * 256.0
    coeffs[0] += a12 * 10321920.0
    coeffs[2] += a12 * (-34406400.0)
    coeffs[4] += a12 * 34406400.0
    coeffs[6] += a12 * (-12902400.0)
    coeffs[8] += a12 * 2090112.0
    coeffs[10] += a12 * (-147456.0)
    coeffs[12] += a12 * 4096.0
    return coeffs

def _g_function(x, a0, a4, a8):
    """Evaluate g(x) = P(x) * exp(-x^2/2)."""
    coeffs = _poly_coeffs(a0, a4, a8)
    p = np.zeros_like(x)
    for k in range(12, -1, -1):
        p = p * x + coeffs[k]
    return p * np.exp(-x**2 / 2.0)

def _find_zeros(coeffs, x_min=-10, x_max=10, n_grid=10000):
    """Find approximate zeros of the polynomial P(x) in [x_min, x_max]."""
    x = np.linspace(x_min, x_max, n_grid)
    p = np.zeros_like(x)
    for k in range(12, -1, -1):
        p = p * x + coeffs[k]
    # Find sign changes
    signs = np.sign(p)
    # Handle zeros exactly
    zero_mask = (p == 0)
    zeros = []
    for i in range(len(x) - 1):
        if signs[i] != signs[i+1] and signs[i] != 0 and signs[i+1] != 0:
            # Linear interpolation for zero
            x0 = x[i] - p[i] * (x[i+1] - x[i]) / (p[i+1] - p[i])
            zeros.append(x0)
    # Also check exact zeros
    for i in range(len(x)):
        if zero_mask[i]:
            zeros.append(x[i])
    return sorted(set(zeros))

def _l1_norm_accurate(a0, a4, a8):
    """Compute L1 norm of g(x) = P(x)exp(-x^2/2) using adaptive quadrature with breakpoints at zeros."""
    coeffs = _poly_coeffs(a0, a4, a8)
    zeros = _find_zeros(coeffs)
    
    def g_scalar(x):
        p = 0.0
        for k in range(12, -1, -1):
            p = p * x + coeffs[k]
        return abs(p * np.exp(-x**2 / 2.0))
    
    # Build integration intervals
    breakpoints = [-15.0] + zeros + [15.0]
    breakpoints = sorted(set(breakpoints))
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a = breakpoints[i]
        b = breakpoints[i+1]
        if b - a < 1e-12:
            continue
        try:
            val, err = quad(g_scalar, a, b, limit=200, epsabs=1e-12, epsrel=1e-12)
            total += val
        except Exception:
            # Fallback to numerical integration
            x = np.linspace(a, b, 1000)
            p = np.zeros_like(x)
            for k in range(12, -1, -1):
                p = p * x + coeffs[k]
            g = np.abs(p * np.exp(-x**2 / 2.0))
            if hasattr(np, 'trapezoid'):
                total += np.trapezoid(g, x)
            else:
                total += np.trapz(g, x)
    return total

def _objective(a0, a4, a8):
    """Compute A(g)*A(ĝ) = (1/sqrt(2*pi)) * ||g||_1^2 for the given coefficients."""
    l1 = _l1_norm_accurate(a0, a4, a8)
    if not np.isfinite(l1) or l1 < 0:
        return 1e10
    return (1.0 / np.sqrt(2.0 * np.pi)) * l1 * l1

def construct():
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    best_result = seed_result.copy()
    best_bound = _objective(*seed_result)
    
    # Differential evolution for global search with tighter bounds
    bounds = [(-2.0, -0.5), (0.01, 0.08), (0.00005, 0.001)]
    
    if time.time() < _DEADLINE - 100:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-14,
                polish=True,
                init='sobol',
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Multi-start Nelder-Mead with more starts near the seed
    starts = [
        seed_result,
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.10, 0.038, 0.00030]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.035, 0.00028]),
        np.array([-1.20, 0.045, 0.00035]),
        np.array([-1.00, 0.030, 0.00025]),
        np.array([-1.25, 0.050, 0.00040]),
        np.array([-0.95, 0.025, 0.00020]),
        np.array([-1.30, 0.055, 0.00045]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B polish
    if time.time() < _DEADLINE - 20:
        try:
            res = minimize(
                _objective,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(
                _objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
