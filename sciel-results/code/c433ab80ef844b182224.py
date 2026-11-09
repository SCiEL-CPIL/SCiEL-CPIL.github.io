"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size and periodic local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Initial population: sample several points to find a good starting region
        candidates = []
        for i in range(20):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))
            if len(candidates) > 1:
                candidates.sort(key=lambda t: t[0])

        # Start from the best candidate
        fx, x = candidates[0]
        best_x, best_f = x.copy(), fx
        calls_used = 20

        # Adaptive (1+1)-ES
        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 10.0
        success_count = 0
        window = 10  # window for success rate tracking

        # CMA-like covariance tracking for better exploration
        # Use a simple diagonal covariance (step sizes per dimension)
        step_sizes = np.ones(dim) * sigma

        for iteration in range(budget - calls_used):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = x + step_sizes * direction
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                success_count += 1
                # Increase step sizes
                step_sizes *= 1.1
                step_sizes = np.clip(step_sizes, sigma_min, sigma_max)
            else:
                # Failure
                step_sizes *= 0.9
                step_sizes = np.clip(step_sizes, sigma_min, sigma_max)

            # Periodic local refinement: every 50 iterations, do a small local search
            if iteration % 50 == 49 and calls_used < budget - 10:
                # Try a few smaller steps
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    small_dir = rng.standard_normal(dim)
                    y2 = np.clip(x + 0.1 * step_sizes * small_dir, -5, 5)
                    fy2 = func(y2)
                    calls_used += 1
                    if fy2 < fx:
                        x, fx = y2, fy2
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()

            # Occasionally reset step sizes if they've become too small
            if np.max(step_sizes) < 0.01 and iteration % 100 == 99:
                step_sizes = np.ones(dim) * 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
