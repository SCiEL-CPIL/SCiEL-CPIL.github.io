"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Multi-start: try several random starting points
        n_starts = 8
        for s in range(n_starts):
            if not time_ok() or calls_made >= self.budget - 10:
                break

            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            # Local search from this starting point
            sigma = 1.5
            for i in range(50):
                if not time_ok() or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.75

        # Final local refinement from best point if we have budget left
        if best_x is not None and time_ok() and calls_made < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            for i in range(20):
                if not time_ok() or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.8

        return best_x, best_fx
# EVOLVE-BLOCK-END
