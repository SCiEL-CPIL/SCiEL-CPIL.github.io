"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with fast 1/5-rule adaptation for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: try several initial points to find both Lunacek basins
        # The Lunacek bi-Rastrigin has a global minimum and a secondary minimum
        # We use diverse starting points to cover the landscape

        # Generate candidate starting points
        # Include points near origin and spread across the domain
        n_starts = 8
        starts = []
        # Center point
        starts.append(np.zeros(dim))
        # Points at various distances from center
        for i in range(1, n_starts):
            scale = 1.0 + i * 0.5
            pt = rng.uniform(-scale, scale, dim)
            starts.append(np.clip(pt, -5, 5))

        # Evaluate all starting points
        best_x = None
        best_f = np.inf
        for s in starts:
            fs = func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        calls_made = n_starts

        # Now do (1+1)-ES from the best starting point with remaining budget
        x = best_x
        fx = best_f
        sigma = 1.0

        # Fast 1/5-rule: increase by 1.2 on success, decrease by 0.8 on failure
        for _ in range(budget - calls_made):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.8

            # If sigma gets too small, do a random restart perturbation
            if sigma < 0.01:
                # Add a moderate random kick to escape potential local minima
                kick = rng.standard_normal(dim) * 0.5
                x = np.clip(x + kick, -5, 5)
                fx = func(x)
                calls_made += 1
                sigma = 0.5

                if time.time() > deadline:
                    break

            # Periodic exploration: every 50 iterations, try a larger step
            if (calls_made - n_starts) % 50 == 0 and (calls_made - n_starts) > 0:
                if time.time() > deadline:
                    break
                explore = np.clip(x + sigma * 2.0 * rng.standard_normal(dim), -5, 5)
                fe = func(explore)
                calls_made += 1
                if fe < fx:
                    x, fx = explore, fe

        return x, fx
# EVOLVE-BLOCK-END
