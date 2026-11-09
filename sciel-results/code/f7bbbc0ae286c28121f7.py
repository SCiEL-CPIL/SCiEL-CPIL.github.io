"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1, num_quad=200):
    """
    Compute the C4 uncertainty upper bound for given double-root locations.
    C4 = (∫₀^∞ |P(x)|² x^(2n-1) e^(-x) dx)^(1/2) / (∫₀^∞ x^(2n-1) e^(-x) dx)^(1/2)
    where P(x) = ∏ᵢ (x - rᵢ)²
    """
    try:
        x_quad, w_quad = np.polynomial.laguerre.laggauss(num_quad)
        
        log_abs_P_sq = np.zeros_like(x_quad)
        for r in roots:
            log_abs_P_sq += 4.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
        alpha = 2 * n - 1
        weight_factor = np.power(x_quad, alpha)
        
        integrand = np.exp(log_abs_P_sq) * weight_factor
        integral_P = np.sum(w_quad * integrand)
        
        log_norm = gammaln(2 * n)
        norm = np.exp(log_norm)
        
        c4_bound = np.sqrt(integral_P / norm)
        
        return c4_bound
    except Exception:
        return np.inf

def _objective(roots_flat, m=12, n=1):
    """Objective function for optimization: the C4 bound."""
    roots = np.asarray(roots_flat)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def _get_initial_roots(m=12, n=1):
    """Get initial root positions using the Laguerre polynomial construction."""
    try:
        degree = 2 * m
        alpha = 2 * n - 1
        L = genlaguerre(degree, alpha)
        roots = L.roots()
        roots = roots[np.real(roots) > 0]
        roots = np.real(roots)
        roots = np.sort(roots)
        if len(roots) >= m:
            return roots[:m]
        else:
            while len(roots) < m:
                roots = np.append(roots, roots[-1] + 1.0)
            return roots
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Uses gradient-based optimization to minimize the C4 uncertainty bound.
    """
    m = 12
    n = 1
    
    # Get initial roots from Laguerre polynomial
    roots_init = _get_initial_roots(m, n)
    roots_init = np.maximum(roots_init, 1e-6)
    roots_init = np.sort(roots_init)
    
    best_roots = roots_init.copy()
    best_score = _objective(roots_init, m, n)
    
    # Try multiple optimization strategies
    strategies = [
        ("Nelder-Mead", "Nelder-Mead"),
        ("L-BFGS-B", "L-BFGS-B"),
    ]
    
    for method_name, method in strategies:
        if _deadline_reached():
            break
        
        try:
            # Use bounds to keep roots positive
            bounds = [(1e-6, None)] * m
            
            result = minimize(
                _objective,
                roots_init,
                args=(m, n),
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < best_score:
                best_roots = result.x
                best_score = result.fun
        except Exception:
            continue
    
    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

def _fallback_construct(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m, alpha=2*n-1)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

# EVOLVE-BLOCK-END
