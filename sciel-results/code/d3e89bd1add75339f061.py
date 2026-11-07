"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Use a subset of budget for multi-start, rest for refinement
        # Strategy: 3 restarts, each with a local search
        n_starts = 3
        calls_per_start = max(10, (budget - n) // n_starts)
        # Reserve some calls for final refinement
        reserve_calls = 50
        calls_per_start = max(10, (budget - reserve_calls) // n_starts)

        best_x = None
        best_f = np.inf

        for start in range(n_starts):
            # Different starting points
            if start == 0:
                x = rng.uniform(-3, 3, n)
            elif start == 1:
                x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-2, 2, n)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive step size local search
            sigma = 1.5 if start == 0 else (1.0 if start == 1 else 0.5)
            step_size = 0.85
            growth = 1.2
            min_sigma = 1e-4
            max_sigma = 4.0
            calls_used = 1

            # Success counter for 1/5 rule
            successes = 0
            total_trials = 0
            window = 5

            while calls_used < calls_per_start and sigma > min_sigma:
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                total_trials += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adaptive step size
                    sigma = min(sigma * growth, max_sigma)
                else:
                    sigma = max(sigma * step_size, min_sigma)

                # Reset window periodically
                if total_trials % window == 0:
                    success_rate = successes / window
                    if success_rate < 0.2:
                        sigma *= 0.7
                    elif success_rate > 0.8:
                        sigma *= 1.3
                    successes = 0

            # Refine with smaller steps if we have budget left
            if calls_used < calls_per_start:
                sigma = max(sigma * 0.5, 0.1)
                while calls_used < calls_per_start and sigma > min_sigma:
                    y = x + sigma * rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = min(sigma * 1.1, max_sigma)
                    else:
                        sigma = max(sigma * 0.9, min_sigma)

        # Final refinement with very small steps around best found
        if best_x is not None and budget - calls_per_start * n_starts > 10:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_per_start * n_starts
            calls_used = 0
            while calls_used < remaining and sigma > 1e-5:
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                    sigma = min(sigma * 1.05, 0.5)
                else:
                    sigma = max(sigma * 0.95, 1e-5)

        return best_x, best_f
# EVOLVE-BLOCK-END
