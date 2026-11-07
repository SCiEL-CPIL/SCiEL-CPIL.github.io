"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Brief global exploration with diverse starts
        # Use a few well-spaced initial points to find a good basin
        best_x = None
        best_f = np.inf
        calls = 0

        n_initial = 5
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            # Deterministic spread along first dim + random elsewhere
            t = (i + 0.5) / n_initial
            x = np.empty(dim)
            x[0] = -5 + 10 * t
            x[1:] = rng.uniform(-5, 5, dim - 1)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local optimization with adaptive (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        successes = 0
        trials = 0
        min_sigma = 0.005
        max_sigma = 4.0
        restart_threshold = 30  # restart after this many failed trials in a row
        failed_streak = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate trial
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                failed_streak = 0
                # Increase sigma on success
                sigma = min(sigma * 1.3, max_sigma)
            else:
                failed_streak += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.85, min_sigma)

            trials += 1

            # Adapt sigma based on success rate over last ~10 trials
            if trials >= 10:
                rate = successes / trials
                if rate > 0.2:
                    sigma = min(sigma * 1.2, max_sigma)
                elif rate < 0.2:
                    sigma = max(sigma * 0.8, min_sigma)
                successes = 0
                trials = 0

            # If stuck (long failed streak and small sigma), try a restart
            if failed_streak > restart_threshold and sigma < 0.1 and calls < budget - 20:
                # Random restart: either local perturbation or global
                if rng.random() < 0.5:
                    x = np.clip(x + 2.0 * rng.standard_normal(dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                sigma = 1.5
                successes = 0
                trials = 0
                failed_streak = 0

        return x, fx
# EVOLVE-BLOCK-END
