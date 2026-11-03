"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_norm_log(roots):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)
    with weight e^{-x}.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound (assuming Bound ~ 1/Norm).
    """
    m = len(roots)
    # Use Gauss-Laguerre quadrature for the integral
    # Need enough nodes to capture the polynomial behavior
    # Roots are up to ~50, so we need nodes up to ~60-80
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log P(x)^2 = 2 * sum log|x - r_i|
    # Handle x == r_i carefully, but quadrature nodes rarely hit exact roots
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        # Avoid log(0)
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    # log_integrand = -x + log_p_sq
    # Sum w * exp(log_integrand)
    # For stability, compute max log_integrand
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    # Sum w * exp(log_integrand - max_log) * exp(max_log)
    # exp(max_log) might overflow, but we return log of the result
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    
    if sum_val <= 0:
        return -np.inf
        
    log_norm_sq = max_log + np.log(sum_val)
    
    return log_norm_sq

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Seed roots: 4, 8, ..., 48
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    best_roots = seed_roots.copy()
    best_log_norm = _compute_norm_log(best_roots)
    
    # We want to MAXIMIZE the norm, so MINIMIZE the negative log norm
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10  # Penalty for timeout
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm = _compute_norm_log(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm # Minimize negative log norm
    
    # Optimization
    try:
        # Start from seed
        result = minimize(objective, seed_roots, method='L-BFGS-B',
                         bounds=[(1e-6, None)] * 12,
                         options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12})
        
        if result.success and result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
            
        # Try a second pass with Nelder-Mead if time permits
        if time.time() - start_time < deadline - 60:
            result2 = minimize(objective, best_roots, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result2.fun < -best_log_norm:
                best_roots = np.sort(result2.x)
                best_log_norm = -result2.fun
                
    except Exception:
        pass
    
    # Final verification: Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
