# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step-size for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def _check_deadline():
            return time.time() < deadline

        # Phase 1: Multi-start exploration to find a good basin
        # Use diverse starting points and short (1+1)-ES runs
        n_starts = 8
        per_start_budget = max(10, (budget // 3) // n_starts)  # ~1/3 of budget for exploration

        starts = []
        # Include origin and some structured points
        starts.append(np.zeros(dim))
        for i in range(1, n_starts):
            starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if not _check_deadline():
                break
            if call_count >= budget:
                break

            x = np.clip(s.copy(), -5, 5)
            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            remaining = min(per_start_budget, budget - call_count)

            for _ in range(remaining):
                if not _check_deadline():
                    break
                if call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.3, 5.0)
                else:
                    sigma *= 0.8

        # Phase 2: Refined (1+1)-ES from the best point found
        if not _check_deadline() and call_count < budget:
            x = best_x.copy() if best_x is not None else rng.uniform(-4, 4, dim)
            if best_x is None:
                fx = func(x)
                call_count += 1
            else:
                fx = best_f

            sigma = 0.5
            remaining = budget - call_count

            for _ in range(remaining):
                if not _check_deadline():
                    break
                if call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.2, 3.0)
                else:
                    sigma *= 0.9

        # Phase 3: Fine-tuning with very small steps
        if not _check_deadline() and call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            remaining = budget - call_count

            for _ in range(remaining):
                if not _check_deadline():
                    break
                if call_count >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
