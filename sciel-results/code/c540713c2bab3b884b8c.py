"""Rewritten optimizer for BBOB-style f14 (Different powers) task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses scipy's Nelder-Mead method (derivative-free) with multiple random restarts.
    This approach avoids gradient estimation issues and is robust to rotations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0
        bounds = [(-5.0, 5.0)] * dim

        # Helper to wrap func and count calls
        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                # Return a dummy value to stop minimize, but don't call func
                return best_f 
            calls_used += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # 1. Initial random scan to find promising basins
        # Use 10% of budget for random search
        num_random = max(10, budget // 10)
        random_points = []
        for _ in range(num_random):
            if calls_used >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f_val = wrapped_func(x)
            random_points.append((f_val, x))
        
        # Sort by fitness and keep the top few as seeds for local search
        random_points.sort(key=lambda x: x[0])
        num_starts = min(3, len(random_points)) # Limit restarts to ensure each has enough budget
        
        # 2. Multi-start Nelder-Mead
        # Allocate remaining budget to the restarts
        remaining_budget = budget - calls_used
        if remaining_budget > 0 and num_starts > 0:
            budget_per_start = remaining_budget // num_starts
            
            for i in range(num_starts):
                if calls_used >= budget - 10: # Safety margin
                    break
                
                x0 = random_points[i][1]
                
                # Nelder-Mead uses ~dim+1 evals per iteration.
                max_iter = max(1, budget_per_start // (dim + 1))
                
                try:
                    res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                except Exception:
                    pass
                
                if calls_used >= budget:
                    break

        # 3. Final random exploration if budget remains
        while calls_used < budget:
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
