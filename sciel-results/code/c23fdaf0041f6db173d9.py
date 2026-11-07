"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initial point: center of the box, which is likely close to the shifted minimum
        # Since the minimizer lies in [-4, 4]^5, starting at 0 is a good heuristic
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 1.0
        calls_made = 1
        
        # Adaptive step size with a wider exploration range
        # Standard (1+1)-ES logic but tuned for Sphere
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Moderate increase
            else:
                sigma *= 0.9 # Moderate decrease
                
            # Clamp sigma to avoid exploding or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
