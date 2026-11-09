import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        history = []
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f if best_f < np.inf else 1e10
            if eval_count >= budget:
                return best_f if best_f < np.inf else 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10
                
            eval_count += 1
            history.append((x_clipped.copy(), f_val))
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Differential Evolution for global search
        # Use a portion of the budget for DE
        de_budget = budget // 2
        de_maxiter = max(1, de_budget // (10 * n))  # Rough estimate of iterations needed
        
        try:
            de_result = differential_evolution(
                wrapped_func,
                bounds=[(-5.0, 5.0)] * n,
                maxiter=de_maxiter,
                popsize=10,
                tol=1e-4,
                seed=42,
                init='latinhypercube'
            )
        except Exception:
            pass
        
        if eval_count >= budget:
            return best_x, best_f
        
        # Phase 2: Local refinement with Nelder-Mead
        # Use the best point found so far
        if best_x is not None:
            try:
                nm_result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass
        
        return best_x, best_f
