"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, multiple restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        dim = self.dim
        budget = self.budget

        # Improved strategy:
        # Phase 1: More aggressive global exploration with shorter, more numerous restarts
        # Phase 2: Focused local refinement with adaptive step size
        # Phase 3: Final high-precision polish near the best point

        # Phase 1: Global exploration - 20 short runs with diverse starting points
        n_restarts = 20
        budget_per_restart = max(1, (budget // 4) // n_restarts)
        remaining = budget

        for restart in range(n_restarts):
            if remaining <= 0:
                break
            if time.time() > deadline:
                break

            # Diverse starting points across the box
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            remaining -= 1

            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.0
            max_steps = budget_per_restart
            consecutive_fail = 0

            for i in range(max_steps):
                if remaining <= 0:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    consecutive_fail = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.8
                    consecutive_fail += 1
                    # If stuck for too long, break early to save budget
                    if consecutive_fail > 20 and i > 15:
                        break

        # Phase 2: Local refinement from the best point found
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            consecutive_fail = 0

            for i in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.08
                    consecutive_fail = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.92
                    consecutive_fail += 1
                    if consecutive_fail > 40:
                        break

        # Phase 3: Final high-precision polish pass
        # Use a very small step size to squeeze out last improvements
        if remaining > 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            consecutive_fail = 0

            for i in range(min(remaining, 50)):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    consecutive_fail = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.95
                    consecutive_fail += 1
                    if consecutive_fail > 20:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
