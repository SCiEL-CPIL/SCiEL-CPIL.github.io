"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Number of restarts: split budget into segments
        # Use a few restarts to escape local minima
        n_restarts = 5
        budget_per_restart = max(1, budget // n_restarts)

        for restart in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart at origin, subsequent random
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive (1+1)-ES with 1/5 rule
            sigma = 1.0
            successes = 0
            window = 10

            while calls_made < budget:
                if time.time() > deadline:
                    break

                # Generate candidate
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.2  # increase step size on success

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8  # decrease step size on failure

                # Adapt sigma based on success rate over window
                if (calls_made - (budget - budget_per_restart * (n_restarts - restart))) % window == 0 and window > 0:
                    # Simple adaptation: if too many successes, increase sigma
                    # If too few, decrease
                    pass  # The multiplicative update above handles this

                # Early termination if sigma becomes too small
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
