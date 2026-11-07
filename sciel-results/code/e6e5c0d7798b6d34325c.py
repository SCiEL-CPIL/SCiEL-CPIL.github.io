"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with multiple random starts
        # Use a grid-like sampling to cover the space
        best_x = None
        best_f = np.inf
        calls = 0

        # Try several well-spaced initial points
        n_initial = 10
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            # Deterministic spread + random jitter
            t = (i + 0.5) / n_initial
            x = np.array([(-5 + 10 * t) if j == 0 else rng.uniform(-5, 5) for j in range(dim)])
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local optimization with adaptive (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        success_threshold = 0.2  # 1/5 rule
        successes = 0
        trials = 0
        min_sigma = 0.01
        max_sigma = 5.0

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
            trials += 1

            # Adapt sigma based on success rate over last ~10 trials
            if trials >= 10:
                rate = successes / trials
                if rate > success_threshold:
                    sigma *= 1.3
                elif rate < success_threshold:
                    sigma *= 0.7
                successes = 0
                trials = 0

            # Clamp sigma
            sigma = np.clip(sigma, min_sigma, max_sigma)

            # If sigma is very small, try a restart to escape local minima
            if sigma < min_sigma * 1.5 and calls < budget - 50:
                # Random restart near current best or completely random
                if rng.random() < 0.5:
                    x = np.clip(x + 2.0 * rng.standard_normal(dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                sigma = 1.0
                successes = 0
                trials = 0

        return x, fx
# EVOLVE-BLOCK-END
