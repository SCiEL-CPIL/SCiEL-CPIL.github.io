"""Portfolio search optimizer for BBOB f20 (Schwefel) with explicit deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size, success-rate control,
    and a final polish phase. Designed for BBOB f20 (Schwefel) on [-5,5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s/330s kill
        dim = self.dim
        total_budget = self.budget

        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def try_func(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return None
            fx = func(x)
            calls_made += 1
            return fx

        # Generate diverse starting points using quasi-random / stratified approach
        n_starts = 10
        starts = []
        for i in range(n_starts):
            local_rng = np.random.default_rng(1000 + i * 137)
            x0 = local_rng.uniform(-4.5, 4.5, dim)
            starts.append(x0)

        # Allocate budget: give more to early starts, less to later ones
        # Reserve ~30 calls for final polish
        polish_budget = 30
        remaining_for_search = total_budget - polish_budget

        # Adaptive budget allocation per start
        # Early starts get more budget
        budget_allocs = []
        remaining = remaining_for_search
        for i in range(n_starts):
            if i < 3:
                alloc = max(30, remaining // 4)
            elif i < 6:
                alloc = max(20, remaining // 8)
            else:
                alloc = max(10, remaining // 12)
            alloc = min(alloc, remaining)
            budget_allocs.append(alloc)
            remaining -= alloc

        for start_idx, x0 in enumerate(starts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            alloc = budget_allocs[start_idx]
            if alloc <= 0:
                continue

            rng = np.random.default_rng(42 + start_idx * 7919)

            x = x0.copy()
            fx = try_func(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            sigma_min = 1e-6
            sigma_max = 5.0

            # Track success rate for adaptive sigma
            successes = 0
            steps = 0
            no_improve = 0

            for _ in range(alloc):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break

                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    sigma = min(sigma * 1.3, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.85, sigma_min)

                # Adaptive sigma based on success rate (1/5 rule inspired)
                if steps >= 10:
                    success_rate = successes / steps
                    if success_rate > 0.35:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif success_rate < 0.1:
                        sigma = max(sigma * 0.9, sigma_min)
                    # Reset counters periodically
                    if steps >= 50:
                        successes = 0
                        steps = 0

                # If no improvement for a while, try a larger jump
                if no_improve > 25:
                    sigma = min(sigma * 2.0, sigma_max)
                    no_improve = 0

        # Final polish: very small sigma around the best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            rng = np.random.default_rng(12345)
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(polish_budget):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = max(sigma * 0.95, 1e-8)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, 1e-8)

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
