"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        total_calls = 0
        budget = self.budget

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Multi-start with (1+1)-ES using adaptive step size
        # Use a larger number of shorter runs to explore the landscape
        n_starts = 10
        # Reserve some budget for final polishing
        polish_budget = max(20, budget // 20)
        search_budget = budget - polish_budget
        budget_per_start = max(10, search_budget // n_starts)

        for start_idx in range(n_starts):
            if check_deadline():
                break

            # Initialize: first at center, then random points
            if start_idx == 0:
                x = np.zeros(self.dim)
            elif start_idx == 1:
                # Try a few deterministic points
                x = rng.uniform(-3, 3, self.dim)
            else:
                x = rng.uniform(-4.5, 4.5, self.dim)

            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.0
            successes = 0
            window = 10
            count = 0
            stall_count = 0
            max_stall = 25

            while count < budget_per_start and not check_deadline():
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
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

                # Adaptive sigma using 1/5 rule
                if count % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.6:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    else:
                        sigma *= 1.05
                    successes = 0

                # Restart if stalled
                if stall_count > max_stall:
                    break

            # If we stalled, try a different initialization for next start
            # by continuing the loop

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

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95

                # Stop if sigma is very small
                if sigma < 1e-7:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
