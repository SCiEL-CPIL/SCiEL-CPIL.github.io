"""Portfolio search for BBOB f7 (Step ellipsoidal) with plateau-aware (1+1)-ES and random restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES with plateau-aware adaptation and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Track global best
        best_x = rng.uniform(-4.0, 4.0, dim)
        best_f = func(best_x)
        calls = 1

        # Plateau-aware (1+1)-ES with restarts
        # Strategy: run short bursts of (1+1)-ES from random starts;
        # if no improvement for a while, restart from a new random point.
        sigma = 1.0
        no_improve = 0
        max_no_improve = 40  # restart after this many steps without improvement

        # Use a schedule: first do a broad random search to find a good region,
        # then refine with (1+1)-ES.

        # Phase 1: random sampling to get a decent initial point (10% of budget)
        n_random = max(10, budget // 10)
        for i in range(n_random):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with restarts for the remaining budget
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1

            if fy < fx - 1e-10:
                # Improvement
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                sigma *= 1.25  # increase step on success
                sigma = min(sigma, 4.0)
            else:
                no_improve += 1
                sigma *= 0.8  # decrease step on failure
                sigma = max(sigma, 0.05)

            # Restart if stuck on plateau for too long
            if no_improve >= max_no_improve:
                # Restart from a new random point, but bias toward current best region
                # Mix of pure random and perturbation of best
                if rng.random() < 0.5:
                    x = rng.uniform(-5.0, 5.0, dim)
                else:
                    x = best_x + 2.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
                no_improve = 0

                # If we've made many restarts, increase aggressiveness
                max_no_improve = max(20, max_no_improve - 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
