"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with portfolio exploration and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng()

        def run_es(x_init, budget, sigma_start=3.0):
            """Run (1+1)-ES for a specific budget."""
            x = x_init.copy()
            fx = func(x)
            calls = 1
            sigma = sigma_start
            min_sigma = 1e-6
            max_sigma = 10.0

            # Aggressive adaptation for fast basin finding
            c_inc = 1.25
            c_dec = 0.8

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * c_inc, max_sigma)
                else:
                    sigma = max(sigma * c_dec, min_sigma)

            return x, fx, calls

        def polish(x_init, budget, sigma_start=0.5):
            """Fine-tuning with smaller sigma and gentler adaptation."""
            x = x_init.copy()
            fx = func(x)
            calls = 1
            sigma = sigma_start
            min_sigma = 1e-8
            max_sigma = 2.0

            c_inc = 1.1
            c_dec = 0.9

            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                if calls >= total_budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)

                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * c_inc, max_sigma)
                else:
                    sigma = max(sigma * c_dec, min_sigma)

            return x, fx, calls

        # Strategy:
        # 1. Explore with 4 random starts, using ~25% of budget each (100 calls).
        # 2. Refine the best found using the remaining ~50% of budget (200 calls).
        
        if total_budget < 20:
            x_init = rng.uniform(-5, 5, dim)
            return run_es(x_init, total_budget)

        n_explore = 4
        # Allocate budget: 4 * 100 = 400 for exploration, 100 for polish
        # But we need to be careful not to exceed total_budget
        budget_explore_each = min(100, total_budget // (n_explore + 1))
        budget_polish = total_budget - n_explore * budget_explore_each
        
        if budget_polish < 10:
            # If budget is tight, do less exploration
            n_explore = 2
            budget_explore_each = total_budget // 3
            budget_polish = total_budget - n_explore * budget_explore_each

        best_x = None
        best_f = float('inf')
        total_calls = 0

        for i in range(n_explore):
            if time.time() > deadline:
                break
            if total_calls >= total_budget:
                break

            x_init = rng.uniform(-5, 5, dim)
            x_run, f_run, calls_used = run_es(x_init, budget_explore_each)
            total_calls += calls_used

            if f_run < best_f:
                best_f = f_run
                best_x = x_run

        # Polish the best point found
        if time.time() < deadline and budget_polish > 0 and best_x is not None:
            # Check if we have enough budget left
            remaining_budget = total_budget - total_calls
            actual_polish_budget = min(budget_polish, remaining_budget)
            if actual_polish_budget > 1:
                x_final, f_final, calls_used = polish(best_x, actual_polish_budget)
                total_calls += calls_used
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final
            else:
                x_final = best_x
                f_final = best_f
        else:
            x_final = best_x
            f_final = best_f

        return x_final, f_final
# EVOLVE-BLOCK-END
