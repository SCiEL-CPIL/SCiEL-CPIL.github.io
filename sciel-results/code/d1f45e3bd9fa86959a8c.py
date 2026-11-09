"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Use multiple restarts to escape local minima
        # Split budget into chunks for each restart
        num_restarts = 10
        budget_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize with a random point, biased towards center for later restarts
            rng = np.random.default_rng(42 + restart * 1000)
            if restart == 0:
                x = rng.uniform(-5, 5, n)
            else:
                # For restarts, sample from a wider distribution but also
                # consider starting near the current best with some perturbation
                if best_x is not None and restart % 2 == 1:
                    # Perturb current best
                    x = np.clip(best_x + rng.normal(0, 2.0, n), -5, 5)
                else:
                    # Fresh random start
                    x = rng.uniform(-5, 5, n)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # Run (1+1)-ES for this restart
            no_improve = 0
            max_no_improve = 50

            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

                # If no improvement for a while, break and restart
                if no_improve > max_no_improve:
                    break

                # Also break if sigma becomes too small or too large
                if sigma < 1e-6 or sigma > 100:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
