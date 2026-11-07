"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with global exploration phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Phase 1: Global Exploration (60% of budget)
        # We divide the budget among multiple independent (1+1)-ES runs.
        # Each run starts from a different random point.
        # This helps escape local minima in f21 (Gallagher 101 peaks).
        
        n_restarts = 10
        calls_per_restart = int((budget * 0.6) / n_restarts)
        
        best_x = None
        best_f = float('inf')
        used_calls = 0

        # Initial grid-like spread to ensure coverage
        # We use a quasi-random sequence (Halton-like via rng) for spread
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Start point: spread across the box
            # Use a combination of random and grid-like structure
            # Simple approach: random points, but ensure some are near boundaries/center
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            used_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local refinement for this start point
            sigma = 2.0
            local_best_x = x.copy()
            local_best_f = fx
            
            for _ in range(calls_per_restart - 1):
                if time.time() > deadline:
                    break
                
                y = np.clip(local_best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used_calls += 1
                
                if fy < local_best_f:
                    local_best_x = y.copy()
                    local_best_f = fy
                    sigma *= 1.1  # Slight increase on success
                else:
                    sigma *= 0.8  # Decrease on failure
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Phase 2: Intensive Local Refinement (40% of budget)
        # Take the best point found and run a high-budget (1+1)-ES on it.
        remaining_budget = budget - used_calls
        if remaining_budget > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Start with moderate step size
            
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                
                if sigma < 1e-6:
                    break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
