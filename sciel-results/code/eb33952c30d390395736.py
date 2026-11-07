"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step sizes and final refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        # Phase 1: Multi-restart (1+1)-ES
        # We'll use a portfolio approach: multiple restarts with different seeds
        # Each restart gets a portion of the budget
        
        best_x = None
        best_f = np.inf
        
        # Number of restarts - use a reasonable number
        # With 500 budget, let's do several restarts
        n_restarts = 8
        budget_per_restart = max(10, (budget - 50) // n_restarts)  # Reserve 50 for final refinement
        
        for restart_idx in range(n_restarts):
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Different starting point for each restart
            rng = np.random.default_rng(42 + restart_idx * 1000)
            
            # For first restart, start near origin (Rastrigin minimum is at 0)
            if restart_idx == 0:
                x = np.zeros(dim)
            else:
                # Start at random points, biased towards center
                x = rng.uniform(-3, 3, dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES with adaptive step size
            sigma = 1.5
            min_sigma = 1e-6
            max_sigma = 5.0
            
            for step in range(budget_per_restart):
                # Check deadline
                if time.time() > deadline - 5.0:
                    break
                
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, min_sigma)
        
        # Phase 2: Fine-grained refinement around global best
        # Use remaining budget for a local search with small steps
        if best_x is not None and time.time() < deadline - 2.0:
            rng = np.random.default_rng(999)
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-8
            max_sigma = 0.5
            
            # Use remaining budget or up to 50 evaluations
            remaining_budget = min(50, budget - (n_restarts * budget_per_restart))
            for step in range(max(1, remaining_budget)):
                if time.time() > deadline - 1.0:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, min_sigma)
        
        # Phase 3: Final very fine-grained local search
        if best_x is not None and time.time() < deadline - 0.5:
            rng = np.random.default_rng(1234)
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            min_sigma = 1e-10
            max_sigma = 0.05
            
            for step in range(20):
                if time.time() > deadline - 0.2:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.7, min_sigma)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
