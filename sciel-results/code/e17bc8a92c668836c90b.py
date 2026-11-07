"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-start, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Multi-start (1+1)-ES
        # Allocate 90% for search, 10% for polish
        search_budget = int(budget * 0.9)
        polish_budget = budget - search_budget

        n_starts = 6
        budget_per_start = max(1, search_budget // n_starts)

        # Starting points: origin, then diverse random points
        start_points = [np.zeros(d)]
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4.5, 4.5, d))

        for start_idx in range(n_starts):
            if check_deadline():
                break

            x = start_points[start_idx].copy()
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # (1+1)-ES with 1/5 success rule
            sigma = 1.5
            successes = 0
            window = 10
            count = 0
            stall_count = 0

            while count < budget_per_start and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                count += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    stall_count += 1

                # 1/5 success rule: adjust sigma every 'window' steps
                if count % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    successes = 0

                # Restart if stalled too much
                if stall_count > 30:
                    break

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                # Stop if sigma is very small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
