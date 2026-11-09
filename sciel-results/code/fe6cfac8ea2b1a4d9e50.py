"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, sigma clamping, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        # Start at the center of the box
        x = np.zeros(self.dim, dtype=np.float64)
        fx = func(x)
        n_calls = 1

        # Initial step size: smaller than the full box width for better early convergence
        sigma = 1.5

        # 1/5 success rule adaptation factors
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        # Sigma bounds to avoid degenerate behavior
        sigma_min = 1e-10
        sigma_max = 20.0

        # Local search acceleration: after finding good progress, reduce exploration
        # and do a fine-tuning phase if budget allows

        for i in range(self.budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break

            # Generate trial point
            direction = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * direction, -5.0, 5.0)
            fy = func(y)
            n_calls += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= c_inc
            else:
                sigma *= c_dec

            # Clamp sigma
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

        return x, fx
# EVOLVE-BLOCK-END
