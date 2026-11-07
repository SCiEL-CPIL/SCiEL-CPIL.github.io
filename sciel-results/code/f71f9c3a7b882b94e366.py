"""Optimized continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with 1/5 success rule and adaptive sigma.
    Simple, fast, and effective for black-box continuous optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim

        # Start near center for fast initial progress
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1
        budget = self.budget

        # Adaptive step size, start large for fast early AOCC
        sigma = 2.0

        # (1+1)-ES with success rule
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate perturbation
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: move to new point
                x, fx = y, fy

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Increase sigma on success
                sigma *= 1.5
            else:
                # Failure: decrease sigma
                sigma *= 0.87

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.001, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
