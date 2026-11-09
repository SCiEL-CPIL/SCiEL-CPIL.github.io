"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Global-scan then multi-restart (1+1)-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan – sample diverse points to identify promising basins
        # Use stratified sampling for better coverage
        n_scan = min(100, budget // 3)
        n_refine_candidates = 3

        # Stratified sampling with finer grid
        n_grid = 3
        grid_points = []
        for i in range(n_grid ** dim):
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % n_grid)
                tmp //= n_grid
            x = np.array([(-4 + 8 * (c + 0.5) / n_grid) + rng.uniform(-0.5, 0.5) * 8 / n_grid for c in coords])
            x = np.clip(x, -5, 5)
            grid_points.append(x)

        # Shuffle and take n_scan points
        idx = rng.permutation(len(grid_points))[:n_scan]
        scan_points = [grid_points[i] for i in idx]

        # Evaluate scan points
        evaluated = []
        for x in scan_points:
            if time.time() > deadline - 5.0:
                break
            fx = func(x)
            evaluated.append((fx, x.copy()))

        if not evaluated:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort by function value, take top candidates
        evaluated.sort(key=lambda t: t[0])
        top_candidates = evaluated[:n_refine_candidates]

        best_fx = top_candidates[0][0]
        best_x = top_candidates[0][1].copy()

        # Phase 2: Refine top candidates with (1+1)-ES
        remaining_budget = budget - len(evaluated)

        # Allocate refinement budget with rank-based weights
        weights = [1.0 / (k + 1) for k in range(n_refine_candidates)]
        total_w = sum(weights)
        refine_budgets = [max(10, int(remaining_budget * w / total_w)) for w in weights]

        for rank, (fx_init, x_init) in enumerate(top_candidates):
            if time.time() > deadline - 10.0:
                break
            if remaining_budget <= 0:
                break

            x = x_init.copy()
            fx = fx_init
            sigma = 1.0

            n_steps = min(refine_budgets[rank], remaining_budget)
            for step in range(n_steps):
                if time.time() > deadline - 2.0:
                    return best_x, best_fx
                if remaining_budget <= 0:
                    break

                # Adaptive sigma with better convergence properties
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15  # More conservative growth
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.8  # Faster decay on failure
                    if sigma < 0.03:
                        sigma = 0.25  # Lower floor, gentler reset

        # Phase 3: Final polish - high-precision local search around best point
        if time.time() < deadline - 3.0 and remaining_budget > 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1

            polish_steps = min(remaining_budget, 30)
            for step in range(polish_steps):
                if time.time() > deadline - 1.0:
                    break

                # Use a very small step for fine-tuning
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9
                    if sigma < 0.005:
                        sigma = 0.02

        return best_x, best_fx
# EVOLVE-BLOCK-END
