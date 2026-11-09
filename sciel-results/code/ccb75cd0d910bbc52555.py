"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multiple short (1+1)-ES runs with restarts
        # Use a smaller budget per run to allow exploration of multiple basins
        runs = 0
        max_runs = 20
        budget_per_run = max(20, total_budget // max_runs)
        no_improve_threshold = 50  # restart if no improvement for this many steps

        for run in range(max_runs):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Initialize: either random or near best
            if run == 0:
                x = rng.uniform(-5, 5, n)
            else:
                # Perturb around best found so far
                x = best_x + rng.normal(0, 1.0, n)
                x = np.clip(x, -5, 5)

            fx = safe_func(x)
            if calls_made >= total_budget:
                break

            sigma = 1.5
            no_improve = 0

            while calls_made < total_budget and no_improve < no_improve_threshold:
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if calls_made >= total_budget:
                    break

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

            runs += 1

        # Phase 2: Final polish with smaller steps around best
        if calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(50):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.95
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
