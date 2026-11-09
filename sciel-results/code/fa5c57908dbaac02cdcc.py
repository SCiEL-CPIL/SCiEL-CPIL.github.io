import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_bound(root_locs):
    """
    Compute a proxy for the C4 upper bound.
    We minimize the L2 norm of the polynomial squared under the Laguerre weight,
    which correlates with tightening the uncertainty bound.
    """
    m = len(root_locs)
    if m == 0:
        return np.inf
    
    # Use Gauss-Laguerre quadrature with sufficient precision
    n_quad = 200
    xq, wq = roots_laguerre(n_quad)
    
    # Calculate P(x) = prod (x - r_i)^2
    # We compute P(x)^2 * e^{-x} for the integral
    # To avoid overflow/underflow, we can compute log-sum-exp or just be careful with scaling
    # For 12 roots, (x-r)^2 can be large.
    
    # Initialize P_sq = 1
    p_sq = np.ones_like(xq)
    for r in root_locs:
        p_sq *= (xq - r)**2
        
    # The integrand is P(x)^2 * e^{-x}
    # Note: roots_laguerre(n_quad) returns nodes and weights for integral int_0^inf f(x) e^{-x} dx
    # So we just sum p_sq * wq
    val = np.sum(p_sq * wq)
    
    if not np.isfinite(val) or val < 0:
        return 1e100
        
    return val

def _objective(params):
    """
    Objective function for the optimizer.
    params: array of 12 root locations.
    """
    # Ensure positivity
    if np.any(params <= 0):
        return 1e100
        
    return _compute_bound(params)

def construct():
    start_time = time.time()
    deadline = start_time + 580.0
    
    m = 12
    
    # Initial guess: Roots of L_12^(1)
    # The directive suggests modeling as zeros of (L_12^(1))^2, which are the same as zeros of L_12^(1)
    try:
        initial_roots = roots_laguerre(m, 1.0).x
        initial_roots = sorted(initial_roots)
    except Exception:
        # Fallback to standard Laguerre
        initial_roots = sorted(roots_laguerre(m).x)
        
    best_roots = initial_roots
    best_val = _compute_bound(best_roots)
    
    # Optimization
    # We optimize the roots directly.
    # Bounds: [1e-6, 100]
    bounds = [(1e-6, 100.0)] * m
    
    try:
        # Use L-BFGS-B for efficiency
        res = minimize(
            _objective,
            initial_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            # Sort the optimized roots to ensure consistent ordering
            optimized_roots = sorted(res.x)
            # Verify positivity
            if np.all(optimized_roots > 0):
                best_roots = optimized_roots
                best_val = res.fun
                
    except Exception:
        pass
        
    # If we have time, try a second phase with a different initial guess
    # e.g., roots of L_12^(0)
    if time.time() < deadline - 10.0:
        try:
            alt_initial = sorted(roots_laguerre(m, 0.0).x)
            res2 = minimize(
                _objective,
                alt_initial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res2.success and np.isfinite(res2.fun) and res2.fun < best_val:
                optimized_roots2 = sorted(res2.x)
                if np.all(optimized_roots2 > 0):
                    best_roots = optimized_roots2
                    best_val = res2.fun
        except Exception:
            pass
            
    # Final check
    if not np.all(np.isfinite(best_roots)) or np.any(best_roots <= 0):
        return sorted(roots_laguerre(m).x)
        
    return list(best_roots)
