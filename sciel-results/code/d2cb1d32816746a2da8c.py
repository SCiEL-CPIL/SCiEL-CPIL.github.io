"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        sigma = 3.0
        min_sigma = 1e-8
        max_sigma = 10.0

        # Aggressive adaptation: increase on success, decrease on failure
        # For unimodal functions, we want to converge fast
        success_factor = 1.7
        failure_factor = 0.75

        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                sigma = min(sigma * success_factor, max_sigma)
            else:
                sigma = max(sigma * failure_factor, min_sigma)

            # If sigma is very small, we're likely converged
            if sigma < 1e-6:
                # Try a few more steps with tiny perturbations
                for _ in range(min(10, self.budget - calls)):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + 1e-5 * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
