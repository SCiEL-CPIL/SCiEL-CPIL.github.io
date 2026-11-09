import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Reserve some budget for polishing
        polish_budget = min(50, budget // 10)
        search_budget = budget - polish_budget
        
        best_x = None
        best_f = float('inf')
        evals_used = 0
        
        start_time = time.time()
        deadline = start_time + 320.0
        
        # Multi-Start (1+1)-ES
        while evals_used < search_budget:
            if time.time() > deadline:
                break
            
            # Initialize a new run
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            evals_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 2.0
            stagnation_count = 0
            max_stagnation = 30 # Restart if stuck for 30 steps
            
            # Inner (1+1)-ES loop
            while evals_used < search_budget:
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals_used += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                    stagnation_count = 0
                else:
                    sigma *= 0.87
                    stagnation_count += 1
                
                if stagnation_count >= max_stagnation:
                    break # Trigger restart in outer loop
        
        # Polish Phase with Nelder-Mead
        remaining = self.budget - evals_used
        if remaining > 10 and best_x is not None and time.time() < deadline:
            nm_budget = remaining
            nm_evals = 0
            def nm_func(x_nm):
                nonlocal nm_evals
                if nm_evals >= nm_budget:
                    return best_f
                x_clipped = np.clip(x_nm, -5, 5)
                val = func(x_clipped)
                nm_evals += 1
                return val

            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxfev': nm_budget, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
