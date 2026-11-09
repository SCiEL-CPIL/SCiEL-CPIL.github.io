# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import root

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Powell's Hybrid Method (scipy.optimize.root)
    with a multi-start strategy.
    """
    x0 = np.array(x0, dtype=float)
    
    call_count = 0
    max_calls_allowed = 450  # Leave some buffer under 500
    
    def F_wrapper(x):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls_allowed:
            return np.array([np.nan, np.nan])
        try:
            res = F(x)
            res = np.array(res, dtype=float)
            if not np.all(np.isfinite(res)):
                return np.array([np.nan, np.nan])
            return res
        except Exception:
            return np.array([np.nan, np.nan])

    def try_solve(x_start):
        nonlocal call_count
        try:
            sol = root(F_wrapper, x_start, method='hybr', tol=1e-12, maxiter=50)
            if sol.success:
                res = F_wrapper(sol.x)
                if np.all(np.isfinite(res)):
                    norm_res = np.linalg.norm(res)
                    if norm_res <= 1e-8:
                        return sol.x, norm_res
            # Even if not success, check the final point
            res = F_wrapper(sol.x)
            if np.all(np.isfinite(res)):
                norm_res = np.linalg.norm(res)
                return sol.x, norm_res
            return sol.x, np.inf
        except Exception:
            return x_start, np.inf

    best_x = x0
    best_norm = np.inf
    
    # 1. Try x0
    x_sol, norm_sol = try_solve(x0)
    if norm_sol < best_norm:
        best_x = x_sol
        best_norm = norm_sol
    if norm_sol <= 1e-8:
        return x_sol

    # 2. Multi-start
    # We have used some calls. Let's check remaining budget.
    if call_count > max_calls_allowed - 20:
        return best_x

    scales = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1]
    ]) / np.sqrt(2)
    
    for scale in scales:
        if call_count > max_calls_allowed - 20:
            break
        for d in directions:
            if call_count > max_calls_allowed - 20:
                break
            x_start = x0 + scale * d
            x_sol, norm_sol = try_solve(x_start)
            if norm_sol < best_norm:
                best_x = x_sol
                best_norm = norm_sol
            if norm_sol <= 1e-8:
                return x_sol
                
    return best_x
# EVOLVE-BLOCK-END
