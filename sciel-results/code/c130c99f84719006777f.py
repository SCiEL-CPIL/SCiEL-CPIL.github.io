"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from scipy.special import eval_genlaguerre, roots_genlaguerre

def _compute_c4_bound(roots, m=12, n=1, tq=None, wq=None, L_vals=None):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    The bound is related to the L2 norm of the polynomial with double roots
    under the Laguerre weight w(t) = t^n * exp(-t).
    
    P(t) = prod_i (t - x_i)^2 * L_m^(n)(t)
    C4 ~ 1 / ||P||_{L2(w)}
    """
    x = np.asarray(roots, dtype=np.float64)
    if x.shape != (m,) or np.any(x <= 0) or np.any(np.isnan(x)):
        return 1e10
    
    # Use precomputed quadrature if available, otherwise compute
    if tq is None:
        n_quad = max(300, 6 * m + 50)
        tq, wq = roots_genlaguerre(n_quad, n)
        L_vals = eval_genlaguerre(m, n, tq)
    
    # Evaluate the product prod_i (t - x_i)^2 at quadrature points
    # Use log-sum for numerical stability
    log_prod = np.zeros_like(tq)
    for xi in x:
        log_prod += 2.0 * np.log(np.abs(tq - xi))
    
    prod_sq = np.exp(log_prod)
    
    # P(t) = prod_sq * L_vals
    P_vals = prod_sq * L_vals
    
    # L2 norm squared under Laguerre weight
    norm_sq = np.sum(wq * P_vals ** 2)
    
    if norm_sq <= 0 or np.isnan(norm_sq) or np.isinf(norm_sq):
        return 1e10
    
    # The bound is inversely related to the norm
    return 1.0 / np.sqrt(norm_sq)

def _objective_log(log_roots, m=12, n=1, tq=None, wq=None, L_vals=None):
    """
    Objective function in log-space.
    log_roots: array of log(roots), will be sorted
    """
    log_roots_sorted = np.sort(log_roots)
    roots = np.exp(log_roots_sorted)
    
    if np.any(roots < 1e-10) or np.any(roots > 1e8):
        return 1e10
    
    return _compute_c4_bound(roots, m, n, tq, wq, L_vals)

def _objective_linear(roots, m=12, n=1, tq=None, wq=None, L_vals=None):
    """
    Objective function in linear space.
    """
    roots_sorted = np.sort(np.abs(roots))
    if np.any(roots_sorted < 1e-10) or np.any(roots_sorted > 1e8):
        return 1e10
    return _compute_c4_bound(roots_sorted, m, n, tq, wq, L_vals)

def _coordinate_descent(log_roots, m, n, tq, wq, L_vals, max_iter=200, tol=1e-10):
    """
    Coordinate descent in log-space with adaptive step sizes.
    """
    log_roots = np.sort(log_roots.copy()).astype(np.float64)
    best_val = _objective_log(log_roots, m, n, tq, wq, L_vals)
    
    for outer in range(max_iter):
        improved = False
        for i in range(m):
            # Try to optimize coordinate i using adaptive step sizes
            best_local = best_val
            best_li = log_roots[i]
            
            # Adaptive step sizes
            for scale in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
                for direction in [-1.0, 1.0]:
                    candidate = log_roots.copy()
                    candidate[i] = best_li + direction * scale
                    candidate = np.sort(candidate)
                    
                    # Enforce bounds
                    if np.any(candidate < -15) or np.any(candidate > 15):
                        continue
                    
                    val = _objective_log(candidate, m, n, tq, wq, L_vals)
                    if val < best_local - tol:
                        best_local = val
                        best_li = candidate[i]
                        improved = True
            
            if best_li != log_roots[i]:
                log_roots[i] = best_li
                log_roots = np.sort(log_roots)
                best_val = best_local
        
        if not improved:
            break
    
    return log_roots, best_val

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s limit
    
    # Precompute quadrature
    n_quad = max(300, 6 * m + 50)
    tq, wq = roots_genlaguerre(n_quad, n)
    L_vals = eval_genlaguerre(m, n, tq)
    
    best_score = 1e10
    best_log_roots = None
    
    # Strategy 1: Multi-start with structured initializations + coordinate descent
    structured_inits = [
        # Geometric with various ratios
        np.log(np.array([2.0 ** (i + 1) for i in range(m)])),
        np.log(np.array([1.8 ** (i + 1) for i in range(m)])),
        np.log(np.array([1.5 ** (i + 1) for i in range(m)])),
        np.log(np.array([1.3 ** (i + 1) for i in range(m)])),
        np.log(np.array([1.2 ** (i + 1) for i in range(m)])),
        # Linear spacing
        np.log(np.array([2.0 * (i + 1) for i in range(m)])),
        np.log(np.array([1.0 * (i + 1) for i in range(m)])),
        # Power law
        np.log(np.array([(i + 1) ** 1.5 for i in range(m)])),
        np.log(np.array([(i + 1) ** 2.0 for i in range(m)])),
        # Spread out
        np.log(np.array([0.5 + 3.0 * (i + 1) for i in range(m)])),
        np.log(np.array([1.0 + 2.0 * (i + 1) for i in range(m)])),
    ]
    
    for x0 in structured_inits:
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            log_roots, val = _coordinate_descent(x0, m, n, tq, wq, L_vals, max_iter=100, tol=1e-12)
            if val < best_score:
                best_score = val
                best_log_roots = log_roots
        except Exception:
            continue
    
    # Strategy 2: Random starts + coordinate descent
    rng = np.random.default_rng(42)
    n_random_starts = 30
    
    for _ in range(n_random_starts):
        if time.time() - start_time > deadline - 60:
            break
        
        # Random log-roots in reasonable range
        x0 = rng.uniform(-2, 8, size=m)
        x0 = np.sort(x0)
        
        try:
            log_roots, val = _coordinate_descent(x0, m, n, tq, wq, L_vals, max_iter=150, tol=1e-12)
            if val < best_score:
                best_score = val
                best_log_roots = log_roots
        except Exception:
            continue
    
    # Strategy 3: Nelder-Mead refinement from best
    if time.time() - start_time < deadline - 30 and best_log_roots is not None:
        try:
            def nm_obj(params):
                return _objective_log(params, m, n, tq, wq, L_vals)
            
            res = minimize(nm_obj, best_log_roots, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_log_roots = np.sort(res.x)
        except Exception:
            pass
    
    # Strategy 4: L-BFGS-B refinement from best
    if time.time() - start_time < deadline - 30 and best_log_roots is not None:
        try:
            def lbfgs_obj(params):
                return _objective_log(params, m, n, tq, wq, L_vals)
            
            res = minimize(lbfgs_obj, best_log_roots, method='L-BFGS-B',
                          bounds=[(-15, 15)] * m,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_log_roots = np.sort(res.x)
        except Exception:
            pass
    
    # Strategy 5: Differential evolution if time permits
    if time.time() - start_time < deadline - 120:
        try:
            def de_obj(params):
                return _objective_log(params, m, n, tq, wq, L_vals)
            
            bounds = [(-5, 10)] * m
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=100,
                popsize=12,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                updating='immediate',
                init='latinhypercube',
                time_limit=100.0
            )
            
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_log_roots = np.sort(result_de.x)
        except Exception:
            pass
    
    # Final fallback
    if best_log_roots is None:
        best_log_roots = np.log(np.array([2.0 ** (i + 1) for i in range(m)]))
    
    # Convert to linear space and ensure validity
    best_roots = np.exp(best_log_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
