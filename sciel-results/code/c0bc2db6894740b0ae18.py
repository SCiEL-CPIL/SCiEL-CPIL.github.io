"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0

        # (1+1)-ES loop
        while call_count < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                sigma = min(sigma * 1.2, sigma_max)
                best_x = x.copy()
                best_fx = fx
            else:
                sigma = max(sigma * 0.8, sigma_min)

        # Final polish: smaller sigma, more iterations if time permits
        if call_count < budget:
            sigma = min(sigma, 0.01)
            polish_budget = min(budget - call_count, 50)
            for _ in range(polish_budget):
                if time.time() > deadline - 1.0:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_fx = fx
                else:
                    sigma = max(sigma * 0.7, sigma_min)

        return best_x, best_fx
# EVOLVE-BLOCK-END
