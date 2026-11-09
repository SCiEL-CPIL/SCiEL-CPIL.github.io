"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, factorial, gammaln
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings("ignore")

def _compute_bound(roots):
    """Compute the upper bound on C4 given 12 double-root locations."""
    roots = np.asarray(roots, dtype=np.float64)
    m = 12
    n = 1
    
    if np.any(roots <= 0) or len(roots) != 12:
        return np.inf
    
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-10):
        return np.inf
    
    from math import comb
    energy = comb(24, 12)  # = 2704156
    
    prod_roots = np.prod(roots)
    if prod_roots <= 0:
        return np.inf
    
    bound = np.sqrt(energy) / (prod_roots ** (1.0 / (2.0 * m)))
    
    return bound

def _objective_log(x):
    """Objective in log-space: minimize -log(product) = -sum(log(x_i)).
    Lower objective = higher product = lower bound."""
    roots = np.exp(x)
    if np.any(roots <= 0):
        return 1e10
    # Check for duplicates
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-10):
        return 1e10
    return -np.sum(x)

def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 590
    
    m = 12
    
    # Get Laguerre zeros as initial guess
    try:
        from numpy.polynomial.laguerre import laggauss
        nodes, weights = laggauss(12)
        roots0 = nodes.copy()
    except Exception:
        roots0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Convert to log-space
    log_roots0 = np.log(roots0)
    
    # Define bounds for differential evolution in log-space
    # Roots should be positive; allow a wide range
    # Laguerre zeros of L_12 range from ~1.5 to ~38
    # Allow log range from log(0.1) to log(200)
    log_bounds = [(np.log(0.05), np.log(500.0))] * 12
    
    best_roots = roots0.copy()
    best_bound = _compute_bound(roots0)
    
    # Phase 1: Differential evolution in log-space
    try:
        de_result = differential_evolution(
            _objective_log,
            bounds=log_bounds,
            seed=42,
            maxiter=300,
            popsize=15,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,
            update_interval=1
        )
        
        if de_result.success or de_result.fun < -np.sum(log_roots0):
            log_roots_de = de_result.x
            roots_de = np.exp(log_roots_de)
            roots_de = np.sort(roots_de)
            
            # Verify
            if len(roots_de) == 12 and np.all(roots_de > 0):
                if np.all(np.diff(roots_de) > 1e-10):
                    bound_de = _compute_bound(roots_de)
                    if bound_de < best_bound:
                        best_roots = roots_de
                        best_bound = bound_de
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return list(best_roots)
    
    # Phase 2: Local refinement with L-BFGS-B in log-space
    try:
        res = minimize(
            _objective_log,
            x0=np.log(best_roots),
            method='L-BFGS-B',
            bounds=log_bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if res.fun < -np.sum(np.log(best_roots)):
            log_roots_ref = res.x
            roots_ref = np.exp(log_roots_ref)
            roots_ref = np.sort(roots_ref)
            
            if len(roots_ref) == 12 and np.all(roots_ref > 0):
                if np.all(np.diff(roots_ref) > 1e-10):
                    bound_ref = _compute_bound(roots_ref)
                    if bound_ref < best_bound:
                        best_roots = roots_ref
                        best_bound = bound_ref
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return list(best_roots)
    
    # Phase 3: Try geometric progression optimization
    # For a geometric progression r^i (i=0..11), product = r^(0+1+...+11) = r^66
    # Maximizing product means maximizing r, but roots must be distinct and positive
    # The constraint is that all roots are valid (positive, distinct)
    # Let's optimize the ratio r directly
    def geo_obj(params):
        r = params[0]
        a = params[1]
        roots = a * np.power(r, np.arange(12))
        if np.any(roots <= 0) or np.any(roots > 1e6):
            return 1e10
        if np.any(np.diff(np.sort(roots)) < 1e-10):
            return 1e10
        return -np.sum(np.log(roots))
    
    try:
        geo_res = minimize(
            geo_obj,
            x0=[1.5, 1.0],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        
        if geo_res.fun < -np.sum(np.log(best_roots)):
            r, a = geo_res.x
            roots_geo = a * np.power(r, np.arange(12))
            roots_geo = np.sort(roots_geo)
            
            if len(roots_geo) == 12 and np.all(roots_geo > 0):
                if np.all(np.diff(roots_geo) > 1e-10):
                    bound_geo = _compute_bound(roots_geo)
                    if bound_geo < best_bound:
                        best_roots = roots_geo
                        best_bound = bound_geo
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return list(best_roots)
    
    # Final verification
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = roots0.copy()
    
    return list(best_roots)
# EVOLVE-BLOCK-END
