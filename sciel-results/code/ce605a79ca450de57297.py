"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        calls = 1

        # Adaptation parameters
        sigma_inc = 1.5
        sigma_dec = 0.87
        min_sigma = 1e-12
        max_sigma = 10.0

        # Phase 1: Main ES loop
        while calls < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Adapt
            if fy < fx:
                x = y
                fx = fy
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec

            sigma = np.clip(sigma, min_sigma, max_sigma)

        # Phase 2: Final polish with smaller sigma if time remains
        if time.time() < deadline - 10.0 and calls < self.budget:
            # Try a few more with very small sigma around best
            x = best_x.copy()
            fx = best_fx
            sigma = max(min_sigma, best_fx * 1e-6) if best_fx > 0 else 1e-6
            sigma = min(sigma, 0.1)

            polish_calls = 0
            max_polish = min(50, self.budget - calls)

            for _ in range(max_polish):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                polish_calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7

                sigma = max(sigma, min_sigma)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
