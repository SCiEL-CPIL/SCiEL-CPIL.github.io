import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Sobol + Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        # Phase 1: Global Exploration with Sobol
        # Use 20% of budget for global search
        n_global = max(10, int(self.budget * 0.2))
        
        try:
            sobol = qmc.Sobol(d=self.dim, scramble=True)
            points_unit = sobol.random(n_global)
            # Map to [-4, 4] as the minimum is known to be in this region
            points = points_unit * 8.0 - 4.0
        except Exception:
            # Fallback to random if Sobol fails
            points = np.random.uniform(-4, 4, (n_global, self.dim))

        best_x = None
        best_f = float('inf')
        
        for p in points:
            if time.time() > deadline:
                break
            try:
                f_val = func(p)
                if f_val < best_f:
                    best_f = f_val
                    best_x = p.copy()
            except Exception:
                continue
                
        # Phase 2: Local Refinement with Nelder-Mead
        remaining_budget = self.budget - n_global
        
        if best_x is not None and remaining_budget > 10 and time.time() < deadline:
            count = [0]
            def wrapped_func(x):
                if count[0] >= remaining_budget:
                    return best_f
                if time.time() > deadline:
                    return best_f
                count[0] += 1
                try:
                    return func(x)
                except Exception:
                    return best_f
            
            try:
                # Estimate max iterations
                max_iter = max(10, remaining_budget // (self.dim + 1))
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
