"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start initialization and 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Multi-start to find a good basin quickly
        n_starts = 8
        best_x = None
        best_fx = np.inf

        for i in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        x = best_x
        fx = best_fx
        calls_made = n_starts

        # Phase 2: (1+1)-ES refinement from best starting point
        sigma = 2.5
        min_sigma = 1e-10
        max_sigma = 8.0

        for _ in range(budget - calls_made):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                if sigma > max_sigma:
                    sigma = max_sigma
            else:
                sigma *= 0.85
                if sigma < min_sigma:
                    sigma = min_sigma

        return x, fx
# EVOLVE-BLOCK-END
