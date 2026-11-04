import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    def roots_from_params(params):
        a, b, c = params
        i = np.arange(12, dtype=float)
        return np.exp(a + b * i + c * i * i)
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        r = roots_from_params(params)
        if np.any(r <= 0) or np.any(np.diff(r) <= 1e-10):
            return 1e10
        # Minimize relative spread and encourage concave spacing
        log_r = np.log(r)
        spread = log_r[-1] - log_r[0]
        # Penalty for convex spacing (c > 0)
        curvature_penalty = max(0, params[2]) ** 2
        return spread + 0.1 * curvature_penalty
    
    bounds = [(-5, 5), (0.01, 2), (-0.5, 0.5)]
    
    de_result = differential_evolution(
        objective, bounds, seed=42, maxiter=300, tol=1e-10,
        popsize=20, polish=False, init='sobol'
    )
    
    best_params = de_result.x
    best_value = de_result.fun
    
    if time.time() - start_time < deadline - 30:
        lbfgs_result = minimize(
            objective, best_params, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14}
        )
        if lbfgs_result.fun < best_value:
            best_params = lbfgs_result.x
            best_value = lbfgs_result.fun
    
    r = roots_from_params(best_params)
    
    if time.time() - start_time < deadline - 10:
        def obj_12dim(r):
            if np.any(r <= 0) or np.any(np.diff(r) <= 1e-10):
                return 1e10
            log_r = np.log(r)
            return (log_r[-1] - log_r[0]) + 0.01 * np.sum(np.diff(log_r) ** 2)
        
        polish_result = minimize(
            obj_12dim, r, method='L-BFGS-B',
            bounds=[(1e-4, 1e6)] * 12,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        if polish_result.fun < best_value:
            r = polish_result.x
    
    return list(r)
