"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_fx = func(best_x)
        
        # Track current best
        global_best_x = best_x.copy()
        global_best_fx = best_fx

        # (1+1)-ES parameters
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 5.0
        success_factor = 1.5
        failure_factor = 0.9
        stagnation_limit = 30
        stagnation_count = 0

        calls_made = 1

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < best_fx:
                best_x = y
                best_fx = fy
                sigma = min(sigma * success_factor, max_sigma)
                stagnation_count = 0
                
                if fy < global_best_fx:
                    global_best_fx = fy
                    global_best_x = best_x.copy()
            else:
                sigma = max(sigma * failure_factor, min_sigma)
                stagnation_count += 1

            # Restart logic if stagnated
            if stagnation_count >= stagnation_limit:
                # Try restarting from center or a random point
                # With 500 budget, we can afford a few restarts
                # Let's restart from center with a larger sigma
                best_x = np.zeros(dim)
                best_fx = func(best_x)
                calls_made += 1
                sigma = 2.0
                stagnation_count = 0
                
                if best_fx < global_best_fx:
                    global_best_fx = best_fx
                    global_best_x = best_x.copy()
                    
                if calls_made >= budget:
                    break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
