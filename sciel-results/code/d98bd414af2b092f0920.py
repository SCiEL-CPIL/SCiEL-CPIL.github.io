import numpy as np
import time
from scipy.optimize import fsolve, minimize

def find_root_system(F, x0):
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # Wrapper for F to handle potential errors and count calls? 
    # The task says "each call is counted" by the evaluator. 
    # We don't need to count them ourselves for the evaluator, 
    # but we should avoid infinite loops.
    
    def F_safe(x):
        try:
            res = F(x)
            if not np.all(np.isfinite(res)):
                return np.array([1e10, 1e10])
            return np.array(res, dtype=float)
        except:
            return np.array([1e10, 1e10])

    # Objective for minimization
    def objective(x):
        try:
            fx = F_safe(x)
            return np.dot(fx, fx)
        except:
            return 1e20

    # 1. Try fsolve
    best_x = x0.copy()
    best_norm = np.inf
    
    try:
        sol, info, ier, msg = fsolve(F_safe, x0, full_output=True, maxiter=50)
        if ier == 1:
            norm = np.linalg.norm(F_safe(sol))
            if norm < best_norm:
                best_norm = norm
                best_x = sol
            if norm <= 1e-12:
                return sol
        else:
            # Even if ier != 1, the solution might be good or better than x0
            norm = np.linalg.norm(F_safe(sol))
            if norm < best_norm:
                best_norm = norm
                best_x = sol
    except Exception:
        pass

    # Check time
    if time.time() - start_time > 100:
        return best_x

    # 2. Try Nelder-Mead if fsolve didn't get close enough
    if best_norm > 1e-8:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-16})
            if res.fun < best_norm:
                best_norm = res.fun
                best_x = res.x
            # Check if we actually found a root
            final_norm = np.linalg.norm(F_safe(best_x))
            if final_norm <= 1e-12:
                return best_x
        except Exception:
            pass

    return best_x
