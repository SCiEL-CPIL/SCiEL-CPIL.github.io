"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        best_x = None
        best_f = float('inf')
        calls_made = 0
        deadline = time.time() + 290

        # Parameters for the (1+1)-ES
        sigma_start = 3.0
        sigma_inc = 1.25
        sigma_dec = 0.85
        max_stall = 15
        max_steps_per_restart = 50

        while calls_made < self.budget and time.time() < deadline:
            # Start a new restart
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_x, best_f = x, fx

            sigma = sigma_start
            no_improve = 0
            steps_in_restart = 0
            max_steps = min(max_steps_per_restart, self.budget - calls_made)

            for _ in range(max_steps):
                if calls_made >= self.budget or time.time() >= deadline:
                    break

                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                steps_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= sigma_inc
                    no_improve = 0
                    if fx < best_f:
                        best_x, best_f = x, fx
                else:
                    sigma *= sigma_dec
                    no_improve += 1

                if no_improve >= max_stall or sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
