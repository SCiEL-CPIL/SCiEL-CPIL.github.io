"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        # Start at the center of the box, which is a strong prior for the minimum location
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial step size
        sigma = 2.0
        
        # Adaptation factors for 1/5 success rule
        # Using 1.1 and 1/1.1 for stability
        c_inc = 1.1
        c_dec = 1.0 / 1.1
        
        start_time = time.time()
        # Safety margin: 280s out of 300s limit
        deadline = start_time + 280.0
        
        for i in range(self.budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate candidate
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= c_inc
            else:
                sigma *= c_dec
                
        return x, fx
# EVOLVE-BLOCK-END
