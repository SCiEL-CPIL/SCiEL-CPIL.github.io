"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with larger sigma
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        # 1/5 success rule parameters
        successes = 0
        window = 10
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 5.0

        while calls < budget:
            # Check time deadline
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                # Increase sigma slightly on success
                sigma = min(sigma * 1.05, sigma_max)
            else:
                # Decrease sigma on failure
                sigma = max(sigma * 0.95, sigma_min)

            # Apply 1/5 rule every 'window' steps
            if calls % window == 0:
                if successes >= window / 5.0:
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                successes = 0

        # Phase 2: Final polish with very small steps
        if calls < budget and time.time() < deadline - 10:
            sigma_polish = max(sigma * 0.01, 1e-8)
            while calls < budget:
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.1, 1e-4)
                else:
                    sigma_polish = max(sigma_polish * 0.9, 1e-10)

        return x, fx
# EVOLVE-BLOCK-END
