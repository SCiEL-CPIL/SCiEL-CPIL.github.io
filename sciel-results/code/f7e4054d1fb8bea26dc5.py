"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # We'll do multiple restarts with smaller budgets each
        # First, a quick exploration phase to find a good starting point
        # Then, refine with (1+1)-ES

        # Strategy: Use a portion of budget for random sampling to find good region,
        # then use remaining budget for (1+1)-ES from best point

        # Let's do a hybrid: start with a few random points, pick best, then run ES
        # But we need to be careful about total calls

        # Actually, let's do a simple multi-restart approach:
        # Split budget into chunks, each chunk is a (1+1)-ES run from a random start
        # Keep the best result overall

        num_restarts = 5
        budget_per_restart = total_budget // num_restarts
        remaining = total_budget - budget_per_restart * num_restarts

        best_x = None
        best_f = np.inf

        for r in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break

            # Check budget
            if budget_per_restart <= 0:
                break

            # Random starting point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            budget_per_restart -= 1

            # Run (1+1)-ES for this restart
            sigma = 2.0
            for _ in range(budget_per_restart):
                # Check time
                if time.time() > deadline:
                    return best_x, best_f

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Use remaining budget for last restart
            if r == num_restarts - 1 and remaining > 0:
                # Already counted in budget_per_restart for last iteration
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
