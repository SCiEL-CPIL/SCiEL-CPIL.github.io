"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with diverse initializations for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Multi-restart strategy: try several independent runs
        # Each restart uses a different seed and initial point
        n_restarts = 5
        budget_per_restart = max(50, budget // n_restarts)

        best_x = None
        best_f = np.inf

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline - 5:
                break

            rng = np.random.default_rng(42 + r * 1000)

            # Diverse initializations: mix of random, center, and grid-like points
            if r == 0:
                x = rng.uniform(-5, 5, dim)
            elif r == 1:
                x = np.zeros(dim)
            elif r == 2:
                x = rng.uniform(-3, 3, dim)
            elif r == 3:
                x = rng.choice([-4, -2, 0, 2, 4], size=dim) + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Run (1+1)-ES for this restart
            for i in range(budget_per_restart):
                # Check time
                if time.time() > deadline - 2:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                    # If stagnating, try a larger jump to escape local minimum
                    if sigma < 0.1:
                        sigma = 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
