import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        max_time = 290.0  # Safety margin for 300s limit
        
        # RNG setup
        rng = np.random.default_rng(42)
        
        # Global best
        best_x = None
        best_f = np.inf
        
        # Current search state
        # Initialize with a random point
        current_x = rng.uniform(-5, 5, n)
        current_f = func(current_x)
        calls = 1
        
        if current_f < best_f:
            best_f = current_f
            best_x = current_x.copy()
            
        sigma = 2.0  # Initial step size
        no_improve = 0
        stagnation_limit = 30  # Restart if no improvement for 30 steps
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate candidate
            # Gaussian perturbation
            y = current_x + sigma * rng.standard_normal(n)
            
            # Clip to box [-5, 5]
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = func(y)
            calls += 1
            
            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                
            # Update current state
            if fy < current_f:
                current_x = y
                current_f = fy
                sigma *= 1.1  # Increase step size on success
                no_improve = 0
            else:
                sigma *= 0.9  # Decrease step size on failure
                no_improve += 1
                
            # Stagnation handling: Restart if stuck
            if no_improve > stagnation_limit:
                # Restart from a random point or near best
                # Strategy: Try a random point in the box, or a large step from best
                # Let's try a random point to explore other basins
                current_x = rng.uniform(-5, 5, n)
                # We don't evaluate the restart point immediately to save a call?
                # No, we need its value. But we can just let the next iteration evaluate a step from it?
                # Better: Evaluate the restart point if we have budget.
                if calls < budget:
                    current_f = func(current_x)
                    calls += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                else:
                    # If no budget, just break
                    break
                
                sigma = 3.0  # Large step size for exploration
                no_improve = 0
                
        return best_x, best_f
