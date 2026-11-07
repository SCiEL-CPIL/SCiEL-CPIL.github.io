import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule and stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # ES Parameters
        sigma = 2.0
        window_size = 15
        successes = []
        max_stagnation = 20
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx
        current_x = x.copy()
        current_f = fx
        stagnation = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = current_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            # Update best
            improved = fy < best_f
            if improved:
                best_f = fy
                best_x = y.copy()
                stagnation = 0
            else:
                stagnation += 1
                
            # Update success history for 1/5 rule
            success = fy < current_f
            successes.append(1 if success else 0)
            if len(successes) > window_size:
                successes.pop(0)
            
            # Adapt sigma based on success rate (1/5 rule)
            if len(successes) == window_size:
                rate = sum(successes) / window_size
                if rate > 0.2:
                    sigma *= 1.2
                elif rate < 0.2:
                    sigma /= 1.2
                # else keep same
                
            # Update current position
            if success:
                current_x = y
                current_f = fy
            # else keep current_x unchanged
            
            # Prevent sigma explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Stagnation check: if no global improvement for max_stagnation steps, restart
            if stagnation >= max_stagnation:
                # Check if we have enough budget for a restart (1 call for new start)
                if calls + 1 < budget:
                    current_x = rng.uniform(-5, 5, dim)
                    current_f = func(current_x)
                    calls += 1
                    if current_f < best_f:
                        best_f = current_f
                        best_x = current_x.copy()
                    stagnation = 0
                    sigma = 2.0
                    successes = [] # Reset history
                else:
                    # Not enough budget to restart, increase sigma to try to jump out
                    sigma *= 2.0

        return best_x, best_f
