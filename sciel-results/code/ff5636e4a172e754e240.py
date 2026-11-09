"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre
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
    
    The bound involves computing integrals that can be evaluated via Gauss-Laguerre quadrature.
    For the refined Cohn-Goncalves construction, we compute the sign uncertainty bound.
    """
    try:
        # Get Gauss-Laguerre quadrature nodes and weights
        x_quad, w_quad = np.polynomial.laguerre.laggauss(num_quad)
        
        # The polynomial P(x) with double roots at the given locations
        # P(x) = product over i of (x - r_i)^2
        # For efficiency, compute log|P(x)| to avoid overflow
        log_abs_P = np.zeros_like(x_quad)
        for r in roots:
            log_abs_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
        # The sign uncertainty bound involves:
        # C4 = sup_{x>0} |P(x)| * exp(-x/2) / (some normalization)
        # More precisely, for the Fourier sign uncertainty, we need:
        # The bound is related to the L-infinity norm of P(x) * exp(-x/2)
        
        # Compute |P(x)| * exp(-x/2)
        # Use log to avoid overflow
        log_val = log_abs_P - 0.5 * x_quad
        # Clip to prevent overflow in exp
        log_val = np.clip(log_val, -700, 700)
        val = np.exp(log_val)
        
        # The bound is the maximum of this function
        c4_bound = np.max(val)
        
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
            # Pad with reasonable values
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
    
    # Get initial roots
    roots_init = _get_initial_roots(m, n)
    
    # Try optimization
    best_roots = roots_init.copy()
    best_score = _objective(roots_init)
    
    # Set up bounds: roots must be positive
    bounds = [(1e-6, None)] * m
    
    # Try multiple optimization strategies
    strategies = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}),
        ('Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}),
    ]
    
    for method, options in strategies:
        if _deadline_reached():
            break
        
        try:
            result = minimize(
                _objective,
                roots_init,
                method=method,
                bounds=bounds,
                options=options,
                args=(m, n)
            )
            
            if result.success and result.fun < best_score:
                best_roots = np.asarray(result.x)
                best_score = result.fun
                
                # Check if we should continue optimizing from this point
                if _deadline_reached():
                    break
        except Exception:
            continue
    
    # Final refinement with a tighter optimization if time permits
    if not _deadline_reached() and best_score > 0:
        try:
            result = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10},
                args=(m, n)
            )
            if result.success and result.fun < best_score:
                best_roots = np.asarray(result.x)
                best_score = result.fun
        except Exception:
            pass
    
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
