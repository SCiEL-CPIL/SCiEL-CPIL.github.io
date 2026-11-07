"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and occasional global restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initialize with a few random points to pick a good starting location
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls = 1
        
        # Parameters for the (1+1)-ES
        # 1/5 rule: increase sigma by 1.5 if success rate in last 10 is > 0.2, decrease by 0.87 if < 0.2
        # We'll use a simpler adaptive rule: increase on success, decrease on failure.
        # But the prompt mentions 1/5 success rule in the docstring of the seed.
        # Let's stick to a robust adaptation.
        
        # To improve upon the basic (1+1)-ES, we can introduce:
        # 1. A larger initial sigma or adaptive sigma.
        # 2. Occasional global restarts if stuck (no improvement for a while).
        
        stuck_counter = 0
        max_stuck = 50  # If no improvement for 50 steps, consider a restart
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Check for stuck
            if stuck_counter > max_stuck:
                # Global restart: jump to a new random location
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = func(x_new)
                calls += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = x_new.copy()
                # Keep the current x for local search, or move to the new random point?
                # Moving to the new random point might be better to escape local minima.
                x = x_new
                fx = fx_new
                sigma = 2.0  # Reset sigma
                stuck_counter = 0
                if calls >= self.budget:
                    break
                continue

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                stuck_counter = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
                stuck_counter += 1
                
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
