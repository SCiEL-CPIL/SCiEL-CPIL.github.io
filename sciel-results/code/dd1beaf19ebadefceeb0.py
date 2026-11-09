"""Hybrid (1+1)-ES with CMA-ES-like adaptive steps and restarts for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with CMA-ES-like diagonal covariance and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
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

        rng = np.random.default_rng(42)

        # Initialize with a few random points to get a sense of the landscape
        n_init = min(10, budget)
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Start from the best point found
        center = best_x.copy()
        current_best = best_f

        # CMA-ES-like parameters
        sigma = 1.0  # global step size
        diag = np.ones(dim) * 0.5  # diagonal covariance (axis-aligned)

        # Success rate tracking for sigma adaptation
        window_size = 15
        successes = 0
        total = 0

        # Restart tracking
        consecutive_failures = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate using diagonal covariance
            z = rng.standard_normal(dim)
            candidate = center + sigma * (diag * z)
            candidate = np.clip(candidate, -5.0, 5.0)

            fy = call_func(candidate)

            if fy < current_best:
                # Success: move center toward candidate
                center = candidate.copy()
                current_best = fy
                successes += 1
                consecutive_failures = 0
                # Expand sigma slightly on success
                sigma *= 1.1
                sigma = min(sigma, 5.0)
            else:
                # Failure: shrink sigma
                sigma *= 0.85
                sigma = max(sigma, 0.01)
                consecutive_failures += 1

            total += 1

            # Adapt diagonal covariance based on recent performance
            if total % window_size == 0:
                success_rate = successes / window_size
                if success_rate > 0.6:
                    diag *= 1.15
                    diag = np.clip(diag, 0.05, 5.0)
                elif success_rate < 0.3:
                    diag *= 0.85
                    diag = np.clip(diag, 0.05, 5.0)
                successes = 0
                total = 0

            # Check if we should restart
            # If no improvement for a while, try a new starting point
            if consecutive_failures > 30:
                # Try a few random restarts
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    new_x = rng.uniform(-4, 4, dim)
                    new_fx = call_func(new_x)
                    if new_fx < current_best:
                        center = new_x.copy()
                        current_best = new_fx
                        sigma = 1.0
                        diag = np.ones(dim) * 0.5
                        consecutive_failures = 0
                        successes = 0
                        total = 0
                        break
                else:
                    # No improvement from restarts, reset failure counter
                    consecutive_failures = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
