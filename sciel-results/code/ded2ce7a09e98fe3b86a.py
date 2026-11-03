"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time

_START_TIME = time.time()
_DEADLINE = 620

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = _START_TIME + _DEADLINE
    
    # Use Gauss-Laguerre quadrature for accurate norm computation
    n_quad = 200
    nodes, weights = roots_laguerre(n_quad)
    
    def log_norm_sq(roots):
        """Log of ||L||^2 where L(x) = prod_{i=1}^{12} (x - r_i)^2, weight e^{-x}."""
        # ||L||^2 = int_0^inf prod(x-r_i)^4 e^{-x} dx
        # = sum_i w_i * prod_j (x_i - r_j)^4
        # log = logsumexp_i [ log(w_i) + 4 * sum_j log|x_i - r_j| ]
        log_vals = np.log(weights)  # log weights
        for j in range(len(roots)):
            diff = np.abs(nodes - roots[j])
            diff = np.maximum(diff, 1e-300)
            log_vals += 4.0 * np.log(diff)
        # logsumexp
        m = np.max(log_vals)
        if not np.isfinite(m):
            return -np.inf
        s = np.sum(np.exp(log_vals - m))
        if s <= 0:
            return -np.inf
        return m + np.log(s)
    
    def objective(roots):
        """Negative log norm squared (to minimize)."""
        if np.any(roots <= 0):
            return 1e10
        val = log_norm_sq(roots)
        if not np.isfinite(val):
            return 1e10
        return -val
    
    x0 = 4.0 * np.arange(1, 13, dtype=float)
    bounds = [(0.01, 500.0)] * 12
    
    best_roots = x0.copy()
    best_score = objective(x0)
    
    # Phase 1: Local optimization from multiple starts
    starts = [
        x0,
        2.0 * np.arange(1, 13, dtype=float),
        6.0 * np.arange(1, 13, dtype=float),
        np.geomspace(1.0, 50.0, 12),
        np.linspace(1.0, 30.0, 12),
    ]
    
    for start in starts:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_roots = res.x
        except Exception:
            pass
    
    # Phase 2: Differential Evolution for global exploration
    if time.time() < deadline - 60:
        try:
            def de_obj(roots):
                if time.time() > deadline:
                    return 1e10
                return objective(roots)
            
            de_res = differential_evolution(de_obj, bounds, seed=42,
                                           maxiter=80, popsize=12,
                                           tol=1e-6, polish=False)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_roots = de_res.x
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 10:
        try:
            res_final = minimize(objective, best_roots, method='L-BFGS-B',
                                bounds=[(0.001, 1000.0)] * 12,
                                options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_roots = res_final.x
        except Exception:
            pass
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 0.001)
    
    return [float(r) for r in best_roots]
