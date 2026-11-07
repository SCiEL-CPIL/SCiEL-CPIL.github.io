"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with coarse initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse search to find a good basin
        # Use a fraction of the budget, e.g., 10% or a fixed small number
        n_coarse = min(20, self.budget // 10)
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_coarse):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
                
        # If we used all budget in coarse search (unlikely), return best
        if n_coarse >= self.budget:
            return best_x, best_f

        # Phase 2: Local refinement with (1+1)-ES
        x = best_x
        fx = best_f
        sigma = 1.0
        calls_used = n_coarse
        stagnation_count = 0
        stagnation_limit = 30
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.2  # Increase step size
                stagnation_count = 0
            else:
                sigma *= 0.85  # Decrease step size
                stagnation_count += 1
                
            # Restart if stagnant
            if stagnation_count >= stagnation_limit:
                # Pick a new random start
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls_used += 1
                sigma = 1.0
                stagnation_count = 0
                
                if calls_used >= self.budget:
                    break

        return x, fx
# EVOLVE-BLOCK-END
