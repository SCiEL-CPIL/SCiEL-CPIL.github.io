import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # Phase 1: Global exploration with multiple random starts
        # Use first 30% of budget for exploration
        explore_budget = int(budget * 0.3)
        num_starts = 6
        budget_per_start = max(1, explore_budget // num_starts)
        
        for s in range(num_starts):
            if time.time() > deadline or total_calls >= budget:
                break
                
            x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            # Brief local search from this start
            x = x0.copy()
            sigma = 1.5
            for _ in range(budget_per_start - 1):
                if time.time() > deadline or total_calls >= budget:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                else:
                    # Shrink sigma on failure
                    sigma *= 0.8
                
                sigma = max(0.01, min(sigma, 5.0))
        
        # Phase 2: Intensive local search around best solution
        if best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            successes = 0
            attempts = 0
            window_size = 10
            
            while total_calls < budget:
                if time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                else:
                    # Keep x, don't update
                    pass
                
                attempts += 1
                
                if attempts >= window_size:
                    success_rate = successes / attempts
                    if success_rate > 0.3:
                        sigma *= 1.3
                    elif success_rate < 0.1:
                        sigma *= 0.6
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.001, 3.0)
                    successes = 0
                    attempts = 0
        
        return best_x, best_f
