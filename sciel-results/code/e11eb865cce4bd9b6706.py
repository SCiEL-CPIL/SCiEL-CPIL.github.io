# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma, periodic global exploration, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)

        calls_made = 1
        remaining = self.budget - 1

        # Parameters
        sigma = 2.0  # Start wider for Schwefel's deceptive landscape
        sigma_min = 1e-6
        sigma_max = 15.0
        stall_limit = 30
        stall_count = 0
        exploration_interval = 35  # Periodically explore new region
        exploration_cooldown = 0   # Cooldown after exploration

        while remaining > 0 and time.time() < deadline:
            # Check if we should do a global exploration step
            if exploration_cooldown > 0:
                exploration_cooldown -= 1
            elif stall_count > 0 and (stall_count >= stall_limit or (calls_made % exploration_interval == 0 and stall_count > 10)):
                # Try a new random point
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = func(x_new)
                calls_made += 1
                remaining -= 1

                if fx_new < fx_best:
                    x_best, fx_best = x_new, fx_new
                    sigma = max(sigma * 0.7, sigma_min)
                    stall_count = 0
                    exploration_cooldown = 5
                else:
                    # If exploration didn't help, keep sigma but reset stall
                    stall_count = 0
                    exploration_cooldown = 10
                continue

            # Normal (1+1)-ES step
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma = min(sigma * 1.2, sigma_max)
                stall_count = 0
            else:
                sigma = max(sigma * 0.8, sigma_min)
                stall_count += 1

        # Final high-precision polish pass with very small sigma
        if remaining > 0 and time.time() < deadline:
            sigma_polish = max(sigma * 0.1, 1e-8)
            polish_budget = min(remaining, 50)
            for _ in range(polish_budget):
                if time.time() >= deadline:
                    break
                y = np.clip(x_best + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                remaining -= 1
                if fy < fx_best:
                    x_best, fx_best = y, fy
                    sigma_polish = max(sigma_polish * 0.9, 1e-10)
                else:
                    sigma_polish = max(sigma_polish * 0.95, 1e-10)

        return x_best, fx_best
# EVOLVE-BLOCK-END
