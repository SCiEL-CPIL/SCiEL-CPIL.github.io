"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Initial point: try a few random starts, keep best
        best_x = rng.uniform(-5, 5, dim)
        best_fx = func(best_x)
        calls = 1

        # Try a few more initial points if we have budget
        for _ in range(min(9, budget - 1)):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            calls += 1
            if fx_try < best_fx:
                best_x, best_fx = x_try, fx_try

        if calls >= budget:
            return best_x, best_fx

        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0

        # Success history for adaptive step size
        successes = 0
        total_trials = 0
        history_len = 10
        successes_in_window = 0
        trials_in_window = 0

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                successes_in_window += 1
                # Increase sigma on success
                sigma *= 1.2
            else:
                # Decrease sigma on failure
                sigma *= 0.85

            total_trials += 1
            trials_in_window += 1

            # Adaptive sigma adjustment based on success rate in window
            if trials_in_window >= history_len:
                rate = successes_in_window / trials_in_window
                if rate > 0.6:
                    sigma *= 1.3
                elif rate < 0.2:
                    sigma *= 0.7
                successes_in_window = 0
                trials_in_window = 0

            # Clamp sigma
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # If sigma is very small, try a restart with larger perturbation
            if sigma < 1e-6:
                # Local restart: perturb from best
                x = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                sigma = 1.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
