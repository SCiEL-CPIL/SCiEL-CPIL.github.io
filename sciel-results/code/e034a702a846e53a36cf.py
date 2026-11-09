"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and multi-restart from diverse starts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # Multi-restart (1+1)-ES with diverse initial points
        # The key insight: Rastrigin has many local minima, so we need
        # multiple restarts from diverse points to find the global minimum.
        
        # Use a fixed seed for reproducibility within the instance
        rng = np.random.default_rng(0)
        
        # Number of restarts and budget allocation
        # With 500 total calls, we can do several restarts
        # Each restart gets a portion of the budget
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        best_x = None
        best_f = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Diverse initial points for each restart
            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-4, 4, dim)
            elif restart == 2:
                x = rng.uniform(-4, 4, dim)
            elif restart == 3:
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            fx = func(x)
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES with 1/5 success rule
            sigma = 2.0
            for _ in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
                
                # Keep sigma within reasonable bounds
                if sigma < 1e-4:
                    sigma = 1e-4
                if sigma > 10.0:
                    sigma = 10.0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
