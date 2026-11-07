"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initialization
        # Start at the center of the box, which is often a good heuristic for BBOB functions
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Initial step size
        sigma = 2.0
        
        # 1/5 rule parameters
        # We track the last 15 steps to determine if sigma should increase or decrease
        # Success is defined as improvement in objective value
        window_size = 15
        successes = 0
        steps_in_window = 0
        
        # To make it more robust, we can use a slightly larger window or different threshold
        # Standard 1/5 rule: if success rate > 0.2, increase sigma; else decrease.
        # For dim=5, 15 steps is reasonable.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate a new candidate
            # Normal distribution
            z = rng.standard_normal(dim)
            y = x + sigma * z
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Update 1/5 rule statistics
            if fy < fx:
                # Success
                successes += 1
                x = y
                fx = fy
            else:
                # Failure
                pass
            
            steps_in_window += 1
            
            # If we have enough steps, adjust sigma
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                # If success rate is too high, increase sigma (we are moving too slowly)
                # If success rate is too low, decrease sigma (we are moving too fast/erratically)
                # Target is 1/5 = 0.2
                if success_rate > 0.2:
                    sigma *= 1.5
                else:
                    sigma /= 1.5
                
                # Reset window
                successes = 0
                steps_in_window = 0
            
            # Ensure sigma doesn't become too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)
            
        return x, fx
# EVOLVE-BLOCK-END
