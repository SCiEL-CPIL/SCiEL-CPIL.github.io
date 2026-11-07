"""Optimized BBOB f12 (Bent Cigar) solver with structural changes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Population-based CMA-ES style optimizer with restarts,
    adaptive step size via 1/5 rule, and time-budget guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.time_limit = 320.0  # safety margin before 330s kill

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)

        # Best-so-far tracking
        best_x = np.full(self.dim, np.nan)
        best_f = np.inf
        calls_made = 0

        # Phase 1: Multi-start local search to find a good basin
        # Use a population of candidates with varying scales
        n_starts = min(10, self.budget // 50)
        budget_per_start = max(20, self.budget // (n_starts + 1))

        for s in range(n_starts):
            if calls_made >= self.budget:
                break
            if time.time() - start_time > self.time_limit - 10:
                break

            # Initialize candidate: mix of global and local exploration
            if s == 0:
                # First start: broad random
                x = rng.uniform(-5, 5, self.dim)
            elif s == 1:
                # Second start: center
                x = rng.normal(0, 1, self.dim)
                x = np.clip(x, -5, 5)
            else:
                # Subsequent starts: perturb best found so far
                if not np.isnan(best_f):
                    x = best_x + rng.normal(0, 2.0, self.dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Local (1+1)-ES refinement from this point
            sigma = 1.0
            local_budget = budget_per_start - 1
            for i in range(local_budget):
                if calls_made >= self.budget:
                    break
                if time.time() - start_time > self.time_limit - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adapt sigma based on success rate (coarse)
                if i > 0 and i % 10 == 0:
                    pass  # simplified; sigma adaptation is via accept/reject

            # After local search, update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Fine-tune around the global best with smaller steps
        if calls_made < self.budget and time.time() - start_time < self.time_limit - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            remaining = self.budget - calls_made
            for i in range(remaining):
                if time.time() - start_time > self.time_limit - 2:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
