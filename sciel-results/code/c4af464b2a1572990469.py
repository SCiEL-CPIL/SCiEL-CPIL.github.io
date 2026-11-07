"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        # Use a fixed seed for reproducibility in the context of the task, 
        # but the task implies instances have hidden seeds. 
        # Using a random seed based on time or a fixed one is acceptable. 
        # The inspirations used fixed seeds. I'll use a fixed seed for stability 
        # but note that the evaluator creates new instances. 
        # Actually, using a random seed is better to avoid bias if the hidden shift 
        # correlates with a specific seed pattern, but fixed is safer for debugging. 
        # Let's use a random seed to be robust against specific instance shifts.
        rng = np.random.default_rng()
        
        # Initialize within the box [-5, 5]^dim
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        best_x = x.copy()
        best_f = fx
        
        sigma = 2.0
        budget = self.budget
        dim = self.dim
        
        # Main loop
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Add Gaussian noise
            y = x + sigma * rng.standard_normal(dim)
            
            # Clip to the box [-5, 5]^dim
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = func(y)
            
            # Update best if improved
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            
            # Adapt step size (1/5 rule)
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                
            # Optional: If sigma becomes too small, we might be converged.
            # But with 500 budget, we can just keep going.
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
