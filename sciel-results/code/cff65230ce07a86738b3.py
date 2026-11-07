"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.5

        calls = 1
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Adaptive sigma floor
            sigma = max(sigma, 1e-6)

        # Final polish: try a few small steps around best
        if time.time() < deadline - 5.0:
            sigma_polish = max(sigma * 0.5, 1e-4)
            for _ in range(10):
                if time.time() > deadline:
                    break
                y = best_x + sigma_polish * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x, best_fx = y.copy(), fy

        return best_x, best_fx
# EVOLVE-BLOCK-END
