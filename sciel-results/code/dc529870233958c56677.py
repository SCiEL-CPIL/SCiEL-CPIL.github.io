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
        
        # Number of restarts
        n_restarts = 10
        # Budget per restart
        budget_per_restart = max(10, budget // n_restarts)
        
        best_x = None
        best_f = float('inf')
        
        rng = np.random.default_rng(0)
        
        # Initial center
        x0_center = np.zeros(dim)
        f_center = func(x0_center)
        if f_center < best_f:
            best_f = f_center
            best_x = x0_center.copy()
            
        budget_used = 1
        
        for i in range(n_restarts):
            if (time.time() - start_time) > deadline or budget_used >= budget:
                break
                
            # Generate random start point
            x0 = rng.uniform(-5, 5, dim)
            
            # Run Nelder-Mead
            try:
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxiter': budget_per_restart, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # Estimate budget used
            # Nelder-Mead uses roughly 2 * maxiter evaluations
            budget_used += 2 * budget_per_restart
            
        return best_x, best_f
