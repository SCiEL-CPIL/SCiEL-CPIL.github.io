"""Rewritten program for the suite task; optimized for C4 bound minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_objective(params):
    """
    Compute the log of the ratio: (L2 norm squared of P) / (P(0)^2).
    P(x) = prod (x - r_i)^2.
    We want to MINIMIZE this ratio.
    """
    m = len(params)
    # Ensure positive roots
    if np.any(params <= 1e-8):
        return 1e10
    
    # Compute log |P(0)|
    # P(0) = prod (-r_i). |P(0)| = prod r_i.
    log_p0_sq = 2.0 * np.sum(np.log(params))
    
    # Compute log L2 norm squared
    # Integral of e^{-x} P(x)^2 dx
    # Use Gauss-Laguerre quadrature
    # Roots are positive. Max root likely < 100.
    # 200 nodes is sufficient for high accuracy.
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # log P(x)^2 = 2 * sum log |x - r_i|
    # We compute this vectorized
    # x_quad shape (N,), params shape (M,)
    # diff shape (N, M)
    diff = np.abs(x_quad[:, None] - params[None, :])
    # Avoid log(0)
    diff[diff < 1e-300] = 1e-300
    
    log_p_sq = 2.0 * np.sum(np.log(diff), axis=1)
    
    # Integrand log: -x + log_p_sq
    log_integrand = -x_quad + log_p_sq
    
    # Sum w * exp(log_integrand)
    # Stable log-sum-exp
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return 1e10
        
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    if sum_val <= 0:
        return 1e10
        
    log_norm_sq = max_log + np.log(sum_val)
    
    # Objective: log(NormSq / P0Sq) = log(NormSq) - log(P0Sq)
    return log_norm_sq - log_p0_sq

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    m = 12
    best_roots = None
    best_score = np.inf
    
    # Initial guesses
    # 1. Standard Laguerre roots for L_12^(1)
    # Note: roots_laguerre(m) gives roots of L_m^0.
    # For L_m^n, we can use scipy.special.roots_genlaguerre
    try:
        from scipy.special import roots_genlaguerre
        x, _ = roots_genlaguerre(m, 1)
        seed_roots = x.copy()
    except ImportError:
        # Fallback if genlaguerre not available (unlikely in modern scipy)
        seed_roots = np.linspace(1, 10, m)
        
    initial_guesses = [seed_roots]
    
    # 2. Uniformly spaced
    initial_guesses.append(np.linspace(0.5, 5.0, m))
    
    # 3. Clustered near 1
    initial_guesses.append(np.linspace(0.1, 2.0, m))
    
    # Add random restarts
    np.random.seed(42)
    for _ in range(20):
        # Log-uniform distribution between 0.01 and 50
        r = 10 ** np.random.uniform(np.log10(0.01), np.log10(50), m)
        r.sort()
        initial_guesses.append(r)
        
    bounds = [(1e-6, None)] * m
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B for speed and reliability
            res = minimize(_compute_objective, x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12})
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_roots = res.x.copy()
                
        except Exception:
            pass
            
    # If we still have time, do a more expensive refinement on the best root found
    if best_roots is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(_compute_objective, best_roots, method='Powell',
                            options={'maxiter': 200, 'maxfev': 1000, 'ftol': 1e-14})
            if res2.fun < best_score:
                best_score = res2.fun
                best_roots = res2.x.copy()
        except Exception:
            pass

    # Final check
    if best_roots is None:
        return seed_roots.tolist()
        
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0) or len(best_roots) != m:
        return seed_roots.tolist()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
