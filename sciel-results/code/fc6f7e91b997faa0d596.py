"""Rewritten optimizer for BBOB f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        min_sigma = 1e-10
        max_sigma = 10.0

        # 1/5 success rule: maintain a window of 10 trials
        window = 10
        success_count = 0
        trials = 0
        evals = 1

        while evals < budget:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals += 1
            trials += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            if trials >= window:
                # Adjust sigma based on success rate
                rate = success_count / trials
                if rate > 1/5:
                    sigma *= 1.3
                elif rate < 1/5:
                    sigma *= 0.8
                sigma = max(min_sigma, min(max_sigma, sigma))
                success_count = 0
                trials = 0

        # Final polish: try very small steps around best
        remaining = budget - evals
        if remaining > 0:
            for ts in [1e-3, 1e-4, 1e-5, 1e-6]:
                for _ in range(min(remaining, 20)):
                    if time.time() > deadline or evals >= budget:
                        break
                    y = best_x + ts * rng.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
