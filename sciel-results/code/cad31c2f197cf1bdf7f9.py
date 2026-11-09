"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-4, 4, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        calls = 1

        # 1/5 success rule state
        successes = 0
        total = 0
        # We'll track over a window of 5*dim evaluations for the success rule
        window_size = max(5 * self.dim, 25)
        window = 0

        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                successes += 1
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                pass

            total += 1
            window += 1

            # Adapt sigma every window_size steps
            if window >= window_size:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.05
                elif rate < 0.2:
                    sigma *= 0.95
                successes = 0
                total = 0
                window = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
