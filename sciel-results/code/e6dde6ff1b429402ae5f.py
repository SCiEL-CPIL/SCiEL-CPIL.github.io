import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        # Use a small number of points to find promising basins
        # 20 points is a reasonable trade-off for 5D
        n_explore = min(20, budget // 10) if budget >= 100 else 5
        points = rng.uniform(-5, 5, (n_explore, n))
        
        # Evaluate exploration points
        values = np.zeros(n_explore)
        for i in range(n_explore):
            values[i] = func(points[i])
            
        if time.time() > deadline:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Select top 3 points for exploitation
        top_k = 3
        if n_explore < top_k:
            top_k = n_explore
            
        # Get indices of top_k best points
        top_indices = np.argsort(values)[:top_k]
        
        best_x_global = points[top_indices[0]].copy()
        best_f_global = values[top_indices[0]]
        
        remaining_budget = budget - n_explore
        if remaining_budget <= 0:
            return best_x_global, best_f_global
            
        # Distribute remaining budget among top_k points
        # Ensure at least 1 step per point if possible
        budget_per_es = remaining_budget // top_k
        if budget_per_es < 1:
             # If budget is too small, just return best found
             return best_x_global, best_f_global

        # Phase 2: Local Exploitation with (1+1)-ES
        # We run ES on each of the top_k points
        # To maximize progress, we run them sequentially.
        # The best result from any of these runs is the final result.
        
        for idx in top_indices:
            x = points[idx].copy()
            fx = values[idx]
            
            # If this starting point is worse than the current global best,
            # we might still find a better point nearby, but it's less likely.
            # However, running ES on a worse point can sometimes escape local minima.
            # We'll run it anyway, but track the global best.
            
            sigma = 1.5  # Start with a moderate step size
            
            # Run ES for budget_per_es steps
            for _ in range(budget_per_es):
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                    # Update global best
                    if fx < best_f_global:
                        best_f_global = fx
                        best_x_global = x.copy()
                else:
                    sigma *= 0.87
                    # Prevent sigma from becoming too small or too large
                    if sigma < 1e-6:
                        sigma = 1e-6
                    if sigma > 10.0:
                        sigma = 10.0

        return best_x_global, best_f_global
