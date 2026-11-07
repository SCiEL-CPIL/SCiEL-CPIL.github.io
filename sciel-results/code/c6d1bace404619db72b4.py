import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        dim = self.dim
        budget = self.budget
        
        # State
        x = np.zeros(dim)
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # (1+1)-ES Parameters
        sigma = 2.0
        target_success = 0.2
        window_size = 15
        successes = []
        
        # Phase 1: (1+1)-ES
        # We reserve some budget for Nelder-Mead
        # Let's use 70% of budget for ES, 30% for NM
        es_budget = int(budget * 0.7)
        nm_budget = budget - es_budget
        
        while calls < es_budget:
            if time.time() >= deadline:
                break
                
            # Generate candidate
            z = np.random.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            f_y = func(y)
            calls += 1
            
            # Update success rate
            successes.append(1 if f_y < best_f else 0)
            if len(successes) > window_size:
                successes.pop(0)
                
            # Adapt sigma
            if len(successes) == window_size:
                success_rate = np.mean(successes)
                if success_rate > target_success:
                    sigma *= 1.2
                elif success_rate < target_success:
                    sigma /= 1.2
            
            if f_y < best_f:
                best_f = f_y
                best_x = y.copy()
                x = y.copy()
            else:
                # Keep x same, sigma adapted
                pass
                
        # Phase 2: Nelder-Mead Polish
        if calls < budget:
            # Calculate max iterations for NM
            # NM uses dim + 1 function calls per iteration roughly
            max_iter = max(1, (budget - calls) // (dim + 1))
            
            # Run Nelder-Mead
            # We need to be careful with the callback to count calls
            # But scipy's minimize doesn't have a strict "stop after N calls" option, 
            # only maxiter.
            # If we set maxiter correctly, it should be safe.
            try:
                res = minimize(func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
