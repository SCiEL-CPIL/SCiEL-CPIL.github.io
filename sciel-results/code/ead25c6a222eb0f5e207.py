"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and adaptive step scaling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: stop 15 seconds before the 300s limit
        deadline = start_time + 285.0

        rng = np.random.default_rng(0)
        n = self.dim

        # Initial point: center of the box (often good for Rastrigin due to symmetry)
        # But the problem says minimizer is in [-4, 4]^5, so center is a good start.
        x = np.zeros(n)
        fx = func(x)

        sigma = 1.5
        # Track consecutive failures to adapt sigma more aggressively
        fail_count = 0
        max_fail = 15

        # We have budget-1 iterations remaining
        for i in range(self.budget - 1):
            # Check time before each expensive function call
            if time.time() > deadline:
                break

            # Generate candidate
            # Use a slightly larger step initially, then shrink
            step = sigma * rng.standard_normal(n)
            y = np.clip(x + step, -5, 5)
            
            # Check time again just in case generation was slow (unlikely but safe)
            if time.time() > deadline:
                break

            fy = func(y)

            # Check time after function call
            if time.time() > deadline:
                if fy < fx:
                    x, fx = y, fy
                break

            if fy < fx:
                x, fx = y, fy
                # Success: increase step size moderately
                sigma *= 1.3
                fail_count = 0
            else:
                # Failure: decrease step size
                sigma *= 0.8
                fail_count += 1
                
                # If we've failed many times, try a larger random jump to escape local minima
                # Rastrigin has many local minima, so this is crucial.
                if fail_count >= max_fail:
                    # Perform a larger random restart-like step from current best
                    # Or just a big random step
                    big_step = 2.0 * rng.standard_normal(n)
                    y2 = np.clip(x + big_step, -5, 5)
                    if time.time() < deadline:
                        fy2 = func(y2)
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma = 1.5
                        else:
                            # If even the big step didn't help, shrink sigma significantly
                            sigma *= 0.5
                    fail_count = 0
                    max_fail = 10 # Increase difficulty for next escape attempt

            # Ensure sigma doesn't get too small or too large
            sigma = np.clip(sigma, 0.05, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
