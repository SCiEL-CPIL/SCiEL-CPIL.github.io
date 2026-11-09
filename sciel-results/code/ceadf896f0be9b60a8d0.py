"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step-size and polish phase for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf

        # Known Lunacek basin regions: origin (0) and shifted point
        # The Lunacek bi-Rastrigin function has two basins: one at the origin
        # and one at a shifted location. We'll try both.
        # The standard Lunacek shift is around (2.5, 2.5, ...) or similar.
        # Let's use a few candidate starting points.
        start_points = [
            np.zeros(dim),
            np.full(dim, 2.5),
            np.full(dim, -2.5),
        ]

        # Add a few random starts for diversity
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, dim))

        # Allocate budget: spend some on exploration (multi-start), rest on polishing
        # We'll do a quick evaluation phase from each start, then focus on the best
        explore_budget_per_start = 30
        num_starts = min(len(start_points), total_budget // explore_budget_per_start)

        # Phase 1: Quick exploration from each start point
        candidates = []
        for i in range(num_starts):
            if time.time() > deadline:
                break
            x = np.clip(start_points[i].copy(), -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Refine the top candidates with adaptive (1+1)-ES
        # Focus budget on the best 1-2 candidates
        top_k = min(2, len(candidates))
        remaining_budget = total_budget - sum(1 for _ in range(num_starts))  # approximate
        # Actually, let's track calls more carefully
        calls_made = num_starts
        budget_per_candidate = max(10, (total_budget - calls_made) // top_k)

        for k in range(top_k):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break

            fx, x = candidates[k]
            sigma = 1.5
            # Adaptive (1+1)-ES with 1/5 success rule
            success_count = 0
            trial_count = 0
            window = 10

            for _ in range(budget_per_candidate):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                trial_count += 1

                # 1/5 success rule: adjust sigma every window trials
                if trial_count >= window:
                    if success_count / trial_count > 0.2:
                        sigma *= 1.3
                    elif success_count / trial_count < 0.2:
                        sigma *= 0.7
                    success_count = 0
                    trial_count = 0

        # Phase 3: Polish - fine-grained search around the best point
        if best_x is not None and calls_made < total_budget:
            if time.time() < deadline:
                polish_budget = min(50, total_budget - calls_made)
                x = best_x.copy()
                fx = best_f
                sigma = 0.1
                for _ in range(polish_budget):
                    if time.time() > deadline:
                        break
                    if calls_made >= total_budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.95
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
