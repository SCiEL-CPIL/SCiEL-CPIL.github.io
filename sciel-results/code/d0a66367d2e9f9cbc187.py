"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    The bound is related to the L2 norm of the polynomial with double roots
    under the Laguerre weight w(t) = t^n * exp(-t).
    
    P(t) = prod_i (t - x_i)^2 * L_m^(n)(t)
    C4 ~ 1 / ||P||_{L2(w)}
    """
    x = np.array(roots)
    if np.any(x <= 0) or np.any(np.isnan(x)):
        return 1e10
    
    # Evaluate L_m^(n)(t) at quadrature points
    from scipy.special import eval_genlaguerre, roots_genlaguerre
    
    # Use Gauss-Laguerre quadrature with sufficient points
    # Need enough points to resolve the polynomial of degree 2*m + m = 3m
    n_quad = max(200, 4 * m + 50)
    tq, wq = roots_genlaguerre(n_quad, n)
    
    # Evaluate Laguerre polynomial at quadrature points
    L_vals = eval_genlaguerre(m, n, tq)
    
    # Evaluate the product prod_i (t - x_i)^2 at quadrature points
    # For numerical stability, compute log of absolute value
    prod_sq = np.ones_like(tq)
    for xi in x:
        prod_sq *= (tq - xi) ** 2
    
    # P(t) = prod_sq * L_vals
    P_vals = prod_sq * L_vals
    
    # L2 norm squared under Laguerre weight: integral P(t)^2 * t^n * exp(-t) dt
    # Gauss-Laguerre quadrature: integral f(t) t^n exp(-t) dt ≈ sum wq_i * f(tq_i)
    norm_sq = np.sum(wq * P_vals ** 2)
    
    if norm_sq <= 0 or np.isnan(norm_sq):
        return 1e10
    
    # The bound is inversely related to the norm
    return 1.0 / np.sqrt(norm_sq)

def _exponential_parameterization(params, m=12):
    """
    Parameterize roots as: r_i = exp(a + b*i + c*i^2) for i=0,...,m-1
    This allows for flexible spacing patterns.
    """
    a, b, c = params
    i = np.arange(m, dtype=float)
    log_roots = a + b * i + c * i ** 2
    # Clip to avoid overflow/underflow
    log_roots = np.clip(log_roots, -20, 20)
    roots = np.exp(log_roots)
    # Ensure positive and sorted
    roots = np.sort(roots)
    return roots

def _objective_exp(params, m=12, n=1):
    """Objective function for exponential parameterization."""
    roots = _exponential_parameterization(params, m)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(roots > 1e10):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def _free_objective(params, m=12, n=1):
    """Objective function for free parameter optimization."""
    roots = np.sort(np.abs(params))
    if np.any(roots < 1e-8) or np.any(roots > 1e10):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s limit
    
    best_score = 1e10
    best_roots = None
    
    # Strategy 1: Exponential parameterization with multiple starts
    # The parameters (a, b, c) control the log-spacing pattern
    initial_params = [
        (np.log(2.0), np.log(1.5), 0.0),      # Geometric growth
        (np.log(3.0), np.log(1.3), 0.05),     # Super-geometric
        (np.log(1.0), np.log(1.8), 0.1),      # Strong growth
        (np.log(5.0), np.log(1.1), 0.02),     # Slow growth with slight acceleration
        (np.log(4.0), np.log(1.2), -0.01),    # Decelerating growth
        (np.log(2.5), np.log(1.4), 0.08),     # Moderate super-geometric
        (np.log(6.0), np.log(1.05), 0.0),     # Nearly uniform at high scale
        (np.log(1.5), np.log(2.0), 0.15),     # Very strong growth
    ]
    
    for x0 in initial_params:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            res = minimize(_objective_exp, x0, method='L-BFGS-B',
                          bounds=[(-5, 5), (-2, 3), (-0.5, 0.5)],
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score and res.fun < 1e9:
                best_score = res.fun
                best_roots = _exponential_parameterization(res.x, m)
        except Exception:
            continue
    
    # Strategy 2: Free parameter optimization from best found so far
    if time.time() - start_time < deadline - 30:
        if best_roots is not None:
            x0_free = best_roots.copy()
        else:
            # Fallback initial guess
            x0_free = np.array([2.0 * (i + 1) ** 1.3 for i in range(m)])
        
        try:
            res = minimize(_free_objective, x0_free, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            
            if res.fun < best_score and res.fun < 1e9:
                best_score = res.fun
                best_roots = np.sort(np.abs(res.x))
        except Exception:
            pass
    
    # Strategy 3: Basin hopping - random perturbations + local optimization
    if time.time() - start_time < deadline - 30 and best_roots is not None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            if time.time() - start_time > deadline - 30:
                break
            
            # Perturb the best roots
            perturbation = rng.normal(0, 0.5, size=m)
            perturbed = np.sort(np.maximum(best_roots + perturbation, 0.1))
            
            try:
                res = minimize(_free_objective, perturbed, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                
                if res.fun < best_score and res.fun < 1e9:
                    best_score = res.fun
                    best_roots = np.sort(np.abs(res.x))
            except Exception:
                continue
    
    # Final fallback
    if best_roots is None:
        best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Ensure all roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
