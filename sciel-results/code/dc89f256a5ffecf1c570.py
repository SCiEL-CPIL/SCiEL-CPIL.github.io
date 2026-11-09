"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initialize at origin (often good for BBOB functions with shifts)
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        evals = 1
        
        sigma = 2.0  # Initial step size
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            direction = rng.standard_normal(n)
            x_new = x + sigma * direction
            x_new = np.clip(x_new, -5, 5)
            
            fx_new = func(x_new)
            evals += 1
            
            # Update best
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
            
            # 1/5 success rule: accept if improvement, adapt sigma
            if fx_new < fx:
                # Success
                x = x_new
                fx = fx_new
                sigma *= 1.05  # Increase step size on success
            else:
                # Failure
                sigma *= 0.95  # Decrease step size on failure
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # Periodic restart if stuck (optional)
            # For simplicity, we'll rely on the adaptive sigma

        return best_x, best_f
# EVOLVE-BLOCK-END
