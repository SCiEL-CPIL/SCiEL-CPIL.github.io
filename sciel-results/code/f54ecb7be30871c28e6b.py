"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time-guard, adaptive step-size, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point: center for f6 attractive sector structure
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Phase 1: Exploration with wider initial sigma
        sigma = 4.0
        min_sigma = 0.001
        max_sigma = 6.0
        no_improve = 0
        best_x = x.copy()
        best_fx = fx

        # Track for adaptive sigma: use success-rate based adaptation
        success_window = 25
        successes = []

        while calls < budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                no_improve = 0
                successes.append(1)
                # Adaptive step: increase on success
                sigma = min(sigma * 1.3, max_sigma)
            else:
                no_improve += 1
                successes.append(0)
                # Adaptive step: decrease on failure
                sigma = max(sigma * 0.85, min_sigma)

            # Sliding window success-rate adaptation
            if len(successes) >= success_window:
                rate = np.mean(successes[-success_window:])
                if rate > 0.5:
                    sigma = min(sigma * 1.1, max_sigma)
                elif rate < 0.2:
                    sigma = max(sigma * 0.7, min_sigma)

            # If stuck for a long time, try a small random jump
            if no_improve > 50 and sigma < 0.1:
                jump = 1.0 * rng.standard_normal(dim)
                x = np.clip(x + jump, -5, 5)
                fx = func(x)
                calls += 1
                if calls >= budget:
                    break
                no_improve = 0
                sigma = 0.5

        # Phase 2: Polish near best point with very small steps
        polish_budget = min(50, budget - calls)
        if polish_budget > 0:
            x = best_x.copy()
            fx = best_fx
            sigma = max(sigma * 0.3, 0.01)
            for _ in range(polish_budget):
                if calls >= budget or time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    sigma = max(sigma, 0.0005)

        return best_x, best_fx
# EVOLVE-BLOCK-END
