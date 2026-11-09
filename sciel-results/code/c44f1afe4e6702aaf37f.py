import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.time_limit = 280.0  # Seconds

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = float('inf')
        
        def safe_func(x):
            nonlocal self
            if self.call_count >= self.budget:
                return self.best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            self.call_count += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # 1. Coarse Grid Search
        # 3 points per dimension: -5, 0, 5
        # Total 3^5 = 243 points
        grid_vals = [-5.0, 0.0, 5.0]
        grid_points = np.array(np.meshgrid(*([grid_vals] * self.dim))).T.reshape(-1, self.dim)
        
        grid_results = []
        for p in grid_points:
            if self.call_count >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            f = safe_func(p)
            grid_results.append((f, p))
            
        # Sort by function value
        grid_results.sort(key=lambda x: x[0])
        
        # 2. Multi-Start Nelder-Mead
        # Take top 5 distinct points
        # To ensure diversity, we can just take the top 5, or enforce distance.
        # Top 5 is sufficient for 5D.
        
        num_starts = 5
        starts = [p for f, p in grid_results[:num_starts]]
        
        # Remaining budget
        remaining_budget = self.budget - self.call_count
        if remaining_budget <= 0:
            return self.best_x, self.best_f
            
        budget_per_start = remaining_budget // num_starts
        
        for start_p in starts:
            if self.call_count >= self.budget or time.time() - self.start_time > self.time_limit:
                break
                
            # Use scipy.optimize.minimize with Nelder-Mead
            # We need to limit function calls.
            # scipy doesn't have a direct 'max_func_calls' for Nelder-Mead, 
            # but maxiter controls iterations.
            # Each iteration uses ~1-2 function calls.
            # Let's set maxiter to budget_per_start // 2 to be safe.
            
            max_iter = max(10, budget_per_start // 2)
            
            try:
                res = minimize(safe_func, start_p, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
            except Exception:
                pass

        return self.best_x, self.best_f
