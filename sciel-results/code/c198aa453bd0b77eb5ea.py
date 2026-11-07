"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: LHS global sampling + (1+1)-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: Global Sampling with LHS
        # Use 100 samples to leave more budget for local refinement
        n_samples = 100
        samples = np.zeros((n_samples, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_samples + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points

        f_samples = np.array([func(x) for x in samples])
        calls = n_samples

        if time.time() - start_time > max_time:
            idx = np.argmin(f_samples)
            return samples[idx], f_samples[idx]

        # Sort by fitness
        sorted_idx = np.argsort(f_samples)
        best_x = samples[sorted_idx[0]].copy()
        best_f = f_samples[sorted_idx[0]]

        # Phase 2: Local Refinement with (1+1)-ES on top candidates
        # Use top 5 candidates for better chance of finding the global optimum
        n_candidates = 5
        remaining_budget = budget - calls
        # Reserve 5% of remaining budget for safety
        safety_budget = int(remaining_budget * 0.05)
        alloc_budget = remaining_budget - safety_budget
        budget_per_candidate = max(10, alloc_budget // n_candidates)

        for k in range(n_candidates):
            if calls >= budget or time.time() - start_time > max_time:
                break

            x_start = samples[sorted_idx[k]].copy()
            f_start = f_samples[sorted_idx[k]]

            # (1+1)-ES refinement with adaptive sigma
            x = x_start.copy()
            fx = f_start
            sigma = 0.5
            local_budget = min(budget_per_candidate, budget - calls)

            for _ in range(local_budget):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Final polish with (1+1)-ES from the best point found
        if calls < budget and time.time() - start_time < max_time:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            final_budget = budget - calls

            for _ in range(final_budget):
                if calls >= budget or time.time() - start_time > max_time:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
