"""Optimized search for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based local search with adaptive step sizes and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s out of 300s to be safe
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(123)

        # Initialize population of 10 points
        pop_size = 10
        x_pop = rng.uniform(-5, 5, (pop_size, dim))
        f_pop = np.array([func(x) for x in x_pop])

        best_idx = np.argmin(f_pop)
        x_best = x_pop[best_idx].copy()
        f_best = f_pop[best_idx]
        calls_made = pop_size

        # Per-individual step sizes
        sigma = np.full(pop_size, 2.0)

        # Local search around each population member
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate trial point
                direction = rng.standard_normal(dim)
                trial = x_pop[i] + sigma[i] * direction
                trial = np.clip(trial, -5, 5)

                f_trial = func(trial)
                calls_made += 1

                if f_trial < f_pop[i]:
                    x_pop[i] = trial
                    f_pop[i] = f_trial
                    sigma[i] = min(sigma[i] * 1.3, 4.0)
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = trial.copy()
                else:
                    sigma[i] = max(sigma[i] * 0.8, 0.05)

        return x_best, f_best
# EVOLVE-BLOCK-END
