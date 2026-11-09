"""Hybrid random sampling + (1+1)-ES with 1/5 rule for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid random sampling + (1+1)-ES with 1/5 success rule and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Random sampling to find a good basin
        n_random = min(50, budget)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        # Phase 2: (1+1)-ES with 1/5 rule
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Track success rate for 1/5 rule
        successes = 0
        total_trials = 0

        # Stagnation tracking
        stagnation = 0
        max_stagnation = 50

        # Adaptive sigma bounds
        sigma_min = 1e-6
        sigma_max = 5.0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = call_func(y)

            total_trials += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                successes += 1
                stagnation = 0

                # 1/5 rule: adjust sigma based on success rate
                if total_trials >= 10:
                    rate = successes / total_trials
                    if rate > 0.2:
                        sigma *= 1.1  # too many successes, increase step
                    elif rate < 0.1:
                        sigma *= 0.9  # too few successes, decrease step
                    # Reset counters
                    successes = 0
                    total_trials = 0

                sigma = min(sigma, sigma_max)
            else:
                # Failure
                stagnation += 1

                # If stuck for too long, restart from a random point
                if stagnation > max_stagnation:
                    # Try a fresh random restart
                    new_x = rng.uniform(-4, 4, dim)
                    new_f = call_func(new_x)
                    if new_f < fx:
                        x = new_x
                        fx = new_f
                    else:
                        # Keep current best, reset stagnation
                        pass

                    # Reset sigma and stagnation
                    sigma = 1.0
                    stagnation = 0
                    successes = 0
                    total_trials = 0

            # If sigma becomes too small, consider a restart
            if sigma < sigma_min * 10:
                new_x = rng.uniform(-4, 4, dim)
                new_f = call_func(new_x)
                if new_f < fx:
                    x = new_x
                    fx = new_f
                sigma = 1.0
                stagnation = 0
                successes = 0
                total_trials = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
