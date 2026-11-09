"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with gentle adaptation and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Start at center of the box - strong prior for minimum location
        x = np.zeros(self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= c_inc
            else:
                sigma *= c_dec

        return best_x, best_fx
# EVOLVE-BLOCK-END
