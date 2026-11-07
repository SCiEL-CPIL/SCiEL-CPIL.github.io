import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return 1e10
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            try:
                fx = func(x_clipped)
            except Exception:
                fx = 1e10
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Global Search
        # Use 20% of budget for global random sampling
        global_budget = max(10, budget // 5)
        global_points = rng.uniform(-5, 5, (global_budget, dim))
        
        global_vals = []
        for x in global_points:
            if time.time() > deadline or calls_made >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            global_vals.append((fx, x_clipped))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
        
        if not global_vals:
            # Fallback if something went wrong
            return np.zeros(dim), 0.0

        # Sort by function value
        global_vals.sort(key=lambda t: t[0])
        
        # Phase 2: Local Search from best points
        num_local = 5
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
            
        local_budget = remaining_budget // num_local
        
        if local_budget < 5:
            return best_x, best_f

        for i in range(min(num_local, len(global_vals))):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            start_x = global_vals[i][1]
            
            # Run Nelder-Mead
            try:
                # maxfev limits the calls in this sub-optimization
                res = minimize(wrapper, start_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'maxfev': local_budget, 'xatol': 1e-8, 'fatol': 1e-8})
            except Exception:
                pass

        return best_x, best_f
