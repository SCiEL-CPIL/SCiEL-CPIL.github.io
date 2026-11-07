"""Combined optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and periodic multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf

        # Multi-start: try several random starting points, keep the best trajectory
        n_starts = 3
        calls_made = 0

        for start_idx in range(n_starts):
            # Check budget and time
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # For the first start, use a fixed seed-like position; subsequent starts use random
            if start_idx == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # If we've already spent most of the budget, don't start a new trajectory
            if calls_made > budget - 50:
                break

            # (1+1)-ES with 1/5 success rule on this trajectory
            sigma = 2.0
            sigma_min = 1e-4
            sigma_max = 10.0
            stuck_count = 0
            max_stuck = 100  # restart local search if stuck too long

            while calls_made < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    stuck_count = 0
                    # 1/5 success rule: increase sigma
                    sigma = min(sigma * 1.5, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # 1/5 success rule: decrease sigma
                    sigma = max(sigma * 0.87, sigma_min)
                    stuck_count += 1

                # If stuck too long, break out of this trajectory to try a new start
                if stuck_count >= max_stuck:
                    break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
