"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start (1+1)-ES with adaptive step size and stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        remaining_budget = self.budget
        
        # Parameters for multi-start
        max_restarts = 10
        stagnation_limit = 30  # Restart if no improvement for 30 steps
        min_budget_per_restart = 20  # Don't start a new restart if less than 20 calls left
        
        restart_count = 0
        
        while remaining_budget > min_budget_per_restart and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            # Initialize new restart
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            remaining_budget -= 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            no_improve_count = 0
            current_restart_budget = min(remaining_budget, 100) # Cap single restart length to encourage more restarts if budget allows
            
            for _ in range(current_restart_budget):
                if time.time() > deadline:
                    break
                if remaining_budget <= 0:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining_budget -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve_count = 0
                    sigma *= 1.25
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve_count += 1
                
                # Adapt sigma bounds
                sigma = max(0.01, min(sigma, 5.0))
                
                # Stagnation check
                if no_improve_count >= stagnation_limit:
                    break
            
            restart_count += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
