"""Optimizer for BBOB f6 (Attractive sector): coarse grid + local (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse grid search to locate the basin, then local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def remaining():
            return budget - calls_made

        # --- Phase 1: Coarse grid search over [-5, 5]^dim ---
        # Use 5 levels per dimension: -5, -2.5, 0, 2.5, 5
        # That's 5^5 = 3125 points, but we only have ~300 budget for the grid.
        # Use a smarter approach: sample a subset of grid points.
        # With dim=5, 3^5 = 243 points at levels {-5, 0, 5}
        # Then 4^5 = 1024 at {-5, -1.67, 1.67, 5}
        # Let's use 3 levels first (243 points), then refine.
        grid_levels = [-5.0, 0.0, 5.0]
        grid_points = list(np.meshgrid(*[grid_levels] * dim))
        grid_flat = np.array([np.concatenate([p]) for p in zip(*grid_points)])
        # Shuffle for early progress
        rng.shuffle(grid_flat)

        # Evaluate grid points (use up to ~300 calls)
        grid_budget = min(300, budget)
        for i in range(grid_budget):
            if not time_ok() or calls_made >= budget:
                break
            x = grid_flat[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: Local (1+1)-ES refinement from the best grid point ---
        if best_x is not None and time_ok() and calls_made < budget:
            x = best_x.copy()
            fx = best_f

            # Adaptive sigma with 1/5 success rule
            sigma = 1.0
            successes = 0
            total = 0

            while calls_made < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                total += 1

                # Update sigma every 10 steps
                if total % 10 == 0:
                    success_rate = successes / total
                    if success_rate > 0.2:
                        sigma *= 1.5
                    elif success_rate < 0.2:
                        sigma *= 0.87
                    successes = 0
                    total = 0

                if sigma < 1e-12:
                    break

        if best_x is None:
            # Fallback: should not happen, but just in case
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
