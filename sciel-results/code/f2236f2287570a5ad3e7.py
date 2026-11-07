"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with stagnation-triggered sigma doubling and gentler adaptation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        n = self.budget

        # Initialize near origin (minimizer known to be in [-4,4]^5)
        x = rng.uniform(-2, 2, d)
        fx = func(x)

        sigma = 2.5
        best_x = x.copy()
        best_f = fx
        non_improving = 0

        calls_used = 1

        for step in range(n - 1):
            # Check time budget
            if time.time() > deadline:
                break

            # Stagnation handling: if stuck, increase sigma significantly
            if non_improving >= 30:
                sigma *= 2.0
                non_improving = 0

            x_new = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fx_new = func(x_new)
            calls_used += 1

            # Update best
            if fx_new < best_f:
                best_x = x_new.copy()
                best_f = fx_new

            # Adapt sigma
            if fx_new < fx:
                x, fx = x_new, fx_new
                sigma *= 1.15
                non_improving = 0
            else:
                sigma *= 0.92
                non_improving += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
