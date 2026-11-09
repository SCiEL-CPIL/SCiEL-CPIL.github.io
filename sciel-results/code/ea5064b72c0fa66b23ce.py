"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        calls = 1
        for i in range(self.budget - 1):
            # Check time limit
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma *= 1.5
            else:
                sigma *= 0.87

            # If sigma gets too small, try a restart
            if sigma < 1e-6:
                # Try a few random restarts
                for _ in range(3):
                    xr = rng.uniform(-5, 5, self.dim)
                    fxr = func(xr)
                    calls += 1
                    if fxr < best_fx:
                        best_fx = fxr
                        best_x = xr.copy()
                sigma = 2.0
                x = best_x.copy()
                fx = best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
