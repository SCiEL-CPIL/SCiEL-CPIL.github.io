"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        t_start = time.time()
        deadline = t_start + 280.0  # 20s safety margin before 300s limit

        # Initialize near center since minimizer is in [-4,4]^5
        x = rng.uniform(-1, 1, self.dim)
        fx = func(x)
        sigma = 1.0
        min_sigma = 1e-6

        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.8

            # Clamp sigma to avoid numerical issues
            if sigma < min_sigma:
                sigma = min_sigma
            if sigma > 10.0:
                sigma = 10.0

        return x, fx
# EVOLVE-BLOCK-END
