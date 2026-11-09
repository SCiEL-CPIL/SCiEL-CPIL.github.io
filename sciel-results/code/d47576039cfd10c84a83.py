"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initialize at center of box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_used = 1

        sigma = 2.0
        no_improve = 0
        max_no_improve = 50

        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Accept step
                x, fx = y, fy
                sigma *= 1.5
                no_improve = 0

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Reject step
                sigma *= 0.87
                no_improve += 1

                # Restart if stuck for too long
                if no_improve >= max_no_improve:
                    # Try a new random starting point
                    x_new = rng.uniform(-5, 5, n)
                    fx_new = func(x_new)
                    calls_used += 1

                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()

                    # Reset local search from the better of the two
                    if fx_new < fx:
                        x, fx = x_new, fx_new
                    else:
                        # Keep current x but reset sigma and no_improve
                        sigma = 2.0
                    no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
