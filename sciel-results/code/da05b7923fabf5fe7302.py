"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with stagnation-based restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        best_x = x.copy()
        best_fx = fx
        fail_count = 0
        stagnation_threshold = 30

        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                sigma *= 1.5
                fail_count = 0
                
                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Reject
                sigma *= 0.87
                fail_count += 1
                
                # Stagnation restart
                if fail_count >= stagnation_threshold:
                    # Restart from a new random point
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    sigma = 2.0
                    fail_count = 0
                    
                    # Update best if this random restart is better
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    
                    # If we used an extra call for the restart, 
                    # we effectively consumed 2 calls in this iteration loop step.
                    # The loop counter handles the main iteration, 
                    # but we must ensure we don't exceed budget.
                    # Since we are inside the loop `for _ in range(budget - 1)`,
                    # and we incremented `calls` manually, we should be careful.
                    # However, the loop runs `budget - 1` times.
                    # If we do an extra func call inside, we might exceed budget.
                    # Let's adjust the logic to avoid exceeding budget.
                    # We will break if calls >= budget.
                    if calls >= budget:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
