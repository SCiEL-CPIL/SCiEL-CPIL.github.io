"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Time guard: return best if we exceed 320s (kill is at 330s)
        start_time = time.time()
        deadline = start_time + 320.0

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        no_improve = 0

        for i in range(budget - 1):
            # Check time periodically
            if i % 50 == 0 and time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                no_improve = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.87
                no_improve += 1

            # Restart if stuck for too long
            if no_improve > 50:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                sigma = 2.0
                no_improve = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
