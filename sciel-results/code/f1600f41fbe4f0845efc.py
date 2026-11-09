"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Start at origin - good default for BBOB f14
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        calls_made = 1

        # Success rate tracking for step size adaptation
        successes = 0
        window = 20
        success_count = 0

        for i in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)

            try:
                fy = func(y)
            except Exception:
                break

            calls_made += 1

            # Track success
            if fy < fx:
                x = y
                fx = fy
                success_count += 1

                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx

            # Adapt step size every 'window' steps
            if (i + 1) % window == 0:
                rate = success_count / window
                if rate > 0.3:
                    sigma *= 1.25
                elif rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                success_count = 0

            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
