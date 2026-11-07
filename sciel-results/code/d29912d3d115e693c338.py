import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = None

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        deadline = self.start_time + 295.0
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        
        remaining_budget = self.budget
        
        # Phase 1: Global Initialization
        # Sample 50 points, pick the best
        init_samples = 50
        for _ in range(min(init_samples, remaining_budget)):
            x = rng.uniform(-5, 5, self.dim)
            self.call_count += 1
            fx = func(x)
            remaining_budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if remaining_budget <= 0:
                break
        
        if best_x is None:
             # Fallback if budget was too small (shouldn't happen with 500)
             x = rng.uniform(-5, 5, self.dim)
             self.call_count += 1
             fx = func(x)
             best_f = fx
             best_x = x.copy()
             remaining_budget -= 1

        # Phase 2: Local Refinement with Restarts
        stagnation_limit = 40
        max_iter_per_restart = 100
        
        while remaining_budget > 0 and time.time() < deadline:
            # Start from best_x? Or random?
            # Starting from best_x allows local refinement.
            # If stuck, restart from random.
            # Let's alternate or just start from best_x for the first restart, then random.
            # Actually, if we just finished a stagnation, we are likely stuck in a local min.
            # So restart from a NEW random point is better for global search.
            # But if we just did initialization, we have a good point.
            
            # Strategy:
            # 1. Refine the current best point.
            # 2. If stagnation, pick a new random point and refine that.
            
            # Let's just pick a new random point for every restart to ensure global exploration.
            # But the first restart should probably start from the best found in initialization.
            
            if 'current_x' not in locals():
                 x = best_x.copy()
                 fx = best_f
            else:
                 x = rng.uniform(-5, 5, self.dim)
                 self.call_count += 1
                 fx = func(x)
                 remaining_budget -= 1
                 if fx < best_f:
                     best_f = fx
                     best_x = x.copy()
            
            sigma = 2.0
            no_improve_count = 0
            
            while remaining_budget > 0 and no_improve_count < stagnation_limit and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                self.call_count += 1
                fy = func(y)
                remaining_budget -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve_count += 1
            
            # If we exited due to stagnation, loop continues to next restart (random point)
            # If we exited due to budget/time, loop terminates
            
        return best_x, best_f
