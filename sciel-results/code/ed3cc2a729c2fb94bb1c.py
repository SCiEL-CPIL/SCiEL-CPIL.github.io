"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from scipy.special import eval_genlaguerre, roots_genlaguerre

def _compute_c4_bound(roots, m=12, n=1):
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
    
    # Use Gauss-Laguerre quadrature with sufficient points
    # Polynomial degree is 2m + m = 3m, need at least (3m+1)/2 points for exactness
    # But we're squaring, so degree is 6m, need 3m+1 points minimum
    # Use more for safety
    n_quad = max(300, 6 * m + 50)
    tq, wq = roots_genlaguerre(n_quad, n)
    
    # Evaluate Laguerre polynomial at quadrature points
    L_vals = eval_genlaguerre(m, n, tq)
    
    # Evaluate the product prod_i (t - x_i)^2 at quadrature points
    # Use log-sum-exp style for numerical stability
    log_prod = np.zeros_like(tq)
    for xi in x:
        log_prod += 2.0 * np.log(np.abs(tq - xi))
    
    # Handle sign: (t-xi)^2 is always positive, so sign is +1
    prod_sq = np.exp(log_prod)
    
    # P(t) = prod_sq * L_vals
    P_vals = prod_sq * L_vals
    
    # L2 norm squared under Laguerre weight
    norm_sq = np.sum(wq * P_vals ** 2)
    
    if norm_sq <= 0 or np.isnan(norm_sq) or np.isinf(norm_sq):
        return 1e10
    
    # The bound is inversely related to the norm
    return 1.0 / np.sqrt(norm_sq)

def _objective_logspace(params, m=12, n=1):
    """
    Objective function working in log-space for better conditioning.
    params: log of the roots (must be increasing)
    """
    log_roots = params
    # Enforce monotonicity by taking cumulative max of differences
    # Actually, just sort them - the objective is permutation invariant
    log_roots_sorted = np.sort(log_roots)
    roots = np.exp(log_roots_sorted)
    
    if np.any(roots < 1e-10) or np.any(roots > 1e8):
        return 1e10
    
    return _compute_c4_bound(roots, m, n)

def _objective_linear(params, m=12, n=1):
    """
    Objective function in linear space with sorting.
    """
    roots = np.sort(np.abs(params))
    if np.any(roots < 1e-10) or np.any(roots > 1e8):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 615.0  # safety margin before 630s limit
    
    best_score = 1e10
    best_roots = None
    
    # Strategy 1: Differential evolution in log-space
    # Log-space is better conditioned for root positions spanning orders of magnitude
    bounds_log = [(-5, 10)] * m  # roots from e^-5 to e^10
    
    try:
        result_de = differential_evolution(
            _objective_logspace,
            bounds_log,
            args=(m, n),
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            updating='deferred',
            init='latinhypercube',
            time_limit=120.0
        )
        
        if result_de.fun < best_score and result_de.fun < 1e9:
            best_score = result_de.fun
            best_roots = np.sort(np.exp(result_de.x))
    except Exception:
        pass
    
    # Strategy 2: Local optimization from best found
    if time.time() - start_time < deadline - 60:
        if best_roots is not None:
            x0 = best_roots.copy()
        else:
            # Fallback: geometric spacing
            x0 = np.array([1.5 ** (i + 1) for i in range(m)])
        
        try:
            res = minimize(_objective_linear, x0, method='L-BFGS-B',
                          bounds=[(1e-8, 1e8)] * m,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score and res.fun < 1e9:
                best_score = res.fun
                best_roots = np.sort(np.abs(res.x))
        except Exception:
            pass
    
    # Strategy 3: Basin hopping with smaller perturbations
    if time.time() - start_time < deadline - 30 and best_roots is not None:
        rng = np.random.default_rng(123)
        for _ in range(30):
            if time.time() - start_time > deadline - 30:
                break
            
            # Multiplicative perturbation in log-space
            log_roots = np.log(best_roots)
            perturbation = rng.normal(0, 0.3, size=m)
            perturbed_log = log_roots + perturbation
            perturbed = np.sort(np.exp(perturbed_log))
            
            try:
                res = minimize(_objective_linear, perturbed, method='L-BFGS-B',
                              bounds=[(1e-8, 1e8)] * m,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < best_score and res.fun < 1e9:
                    best_score = res.fun
                    best_roots = np.sort(np.abs(res.x))
            except Exception:
                continue
    
    # Strategy 4: Try a few structured initializations
    if time.time() - start_time < deadline - 30:
        structured_inits = [
            # Geometric with various ratios
            np.array([2.0 ** (i + 1) for i in range(m)]),
            np.array([1.8 ** (i + 1) for i in range(m)]),
            np.array([1.5 ** (i + 1) for i in range(m)]),
            # Linear spacing
            np.array([2.0 * (i + 1) for i in range(m)]),
            np.array([1.0 * (i + 1) for i in range(m)]),
            # Power law
            np.array([(i + 1) ** 1.5 for i in range(m)]),
            np.array([(i + 1) ** 2.0 for i in range(m)]),
        ]
        
        for x0 in structured_inits:
            if time.time() - start_time > deadline - 30:
                break
            
            try:
                res = minimize(_objective_linear, x0, method='L-BFGS-B',
                              bounds=[(1e-8, 1e8)] * m,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < best_score and res.fun < 1e9:
                    best_score = res.fun
                    best_roots = np.sort(np.abs(res.x))
            except Exception:
                continue
    
    # Final fallback
    if best_roots is None:
        best_roots = np.array([2.0 ** (i + 1) for i in range(m)])
    
    # Ensure all roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
