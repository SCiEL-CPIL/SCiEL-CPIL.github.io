"""Rewritten program: (1+1)-ES with aggressive early exploration and adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with large initial step and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        deadline = time.time() + 280  # safety margin before 330s kill

        # Start with a vertex of the box for fast initial gradient capture
        x = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(d)])
        fx = func(x)
        budget_used = 1

        sigma = 5.0  # Large initial step to explore the full box quickly

        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            budget_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.85

            # Floor on sigma to avoid numerical issues
            if sigma < 1e-10:
                sigma = 1e-10

        return x, fx
# EVOLVE-BLOCK-END
