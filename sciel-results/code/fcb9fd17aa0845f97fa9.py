"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive step size and restart mechanism."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initial point - start near center for Weierstrass which has global min near origin
        x = rng.uniform(-1, 1, dim)
        fx = evaluate(x)
        
        sigma = 1.0
        stagnation = 0
        max_stagnation = 40  # Restart if no improvement for 40 steps
        success_count = 0
        window_size = 10
        successes = []
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            
            if fy < fx:
                # Improvement
                x, fx = y, fy
                successes.append(1)
                stagnation = 0
            else:
                # No improvement - keep parent
                successes.append(0)
                stagnation += 1
            
            # Keep window of last window_size successes
            if len(successes) > window_size:
                successes.pop(0)
            
            # 1/5 success rule for step size adaptation
            if len(successes) == window_size:
                rate = sum(successes) / window_size
                if rate > 0.2:
                    sigma *= 1.1  # Increase if success rate too high
                elif rate < 0.1:
                    sigma *= 0.9  # Decrease if success rate too low
                # Otherwise keep sigma
            
            # Restart Mechanism
            if stagnation > max_stagnation:
                # Restart from best found with a larger step size
                x = best_x.copy()
                fx = best_f
                sigma = max(sigma, 1.5)  # Ensure reasonable step size for exploration
                stagnation = 0
                successes = []
        
        return best_x, best_f
# EVOLVE-BLOCK-END
