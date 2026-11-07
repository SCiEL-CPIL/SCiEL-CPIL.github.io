"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, centered start, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Start at center, which is closer to the known minimizer region [-4,4]^5
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        calls_made = 1

        # Adaptive step size with conservative multipliers
        sigma_up = 1.2
        sigma_down = 0.9
        sigma_min = 1e-6
        sigma_max = 10.0

        # Phase 1: (1+1)-ES with adaptive sigma
        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = min(sigma * sigma_up, sigma_max)
            else:
                sigma = max(sigma * sigma_down, sigma_min)

        # Phase 2: Local refinement around best point with smaller steps
        if calls_made < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = max(sigma * 0.1, 1e-5)
            while calls_made < self.budget:
                if time.time() > deadline:
                    break

                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, sigma_max)
                else:
                    sigma = max(sigma * 0.95, 1e-7)
                    if sigma <= 1e-7:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
