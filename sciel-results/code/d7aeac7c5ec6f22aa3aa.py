"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim
        
        # Use multiple restarts to escape local minima
        max_restarts = 10
        calls_per_restart = max(1, budget // max_restarts)
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize with random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            sigma = 2.0
            
            # (1+1)-ES with 1/5 success rule
            for step in range(calls_per_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.5, 10.0)
                else:
                    sigma = max(sigma * 0.87, 0.01)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # If we found a very good solution, we can stop early
            if best_f < -50:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
