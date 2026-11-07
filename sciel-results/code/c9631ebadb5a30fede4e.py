import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        d = self.dim

        # Initialize with a few random starts, pick the best
        n_starts = 10
        best_x = None
        best_f = np.inf

        for i in range(n_starts):
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        x = best_x
        fx = best_f
        sigma = 1.0

        # (1+1)-ES loop
        for _ in range(self.budget - n_starts):
            if time.time() - start_time > timeout:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.95

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
