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
        # Initialize at origin, which is the standard center for BBOB
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls = 1
        
        # Current search state
        current_x = best_x.copy()
        current_f = best_f
        
        sigma = 2.0  # Initial step size
        successes = []
        window_size = 10
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate candidate
            y = current_x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = func(y)
            calls += 1
            
            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                
            # Update current state and record success
            if fy < current_f:
                current_x = y
                current_f = fy
                successes.append(1)
            else:
                successes.append(0)
                
            # Maintain sliding window
            if len(successes) > window_size:
                successes.pop(0)
                
            # Adapt sigma using 1/5 rule
            if len(successes) == window_size:
                success_rate = np.mean(successes)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.15:
                    sigma /= 1.1
                    
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)
            
        return best_x, best_f
