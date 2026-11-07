"""Combined optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and final fine-tuning polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        # Main (1+1)-ES phase: use ~94% of budget for exploration
        main_budget = int(budget * 0.94)
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 10.0

        while calls < main_budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.5, sigma_max)
            else:
                sigma = max(sigma * 0.87, sigma_min)

        # Final fine-tuning polish: use remaining budget with very small sigma
        # to achieve high precision near the optimum
        polish_budget = budget - calls
        if polish_budget > 0 and time.time() < deadline:
            sigma_polish = max(sigma * 0.1, 1e-4)
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.3, 0.5)
                else:
                    sigma_polish = max(sigma_polish * 0.8, 1e-8)

        return x, fx
# EVOLVE-BLOCK-END
