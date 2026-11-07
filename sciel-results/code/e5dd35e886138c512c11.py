"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-aware optimizer: estimate gradient, march to best corner, then refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        def time_ok():
            return time.time() < deadline

        x_best = np.zeros(dim)
        fx_best = func(x_best)

        # Phase 1: Estimate gradient using central differences at origin
        # For a linear function f(x) = c + g·x, this gives exact gradient
        h = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            xp = x_best.copy()
            xm = x_best.copy()
            xp[i] += h
            xm[i] -= h
            fxp = func(xp)
            fxm = func(xm)
            grad[i] = (fxp - fxm) / (2 * h)
            if fxp < fx_best:
                x_best, fx_best = xp, fxp
            if fxm < fx_best:
                x_best, fx_best = xm, fxm

        # Phase 2: For linear function, minimum is at a corner.
        # If grad[i] > 0, set x[i] = -5; if grad[i] < 0, set x[i] = +5
        corner = np.where(grad > 0, -5.0, 5.0)
        fc = func(corner)
        if fc < fx_best:
            x_best, fx_best = corner, fc

        # Also check the opposite corner (in case of numerical issues)
        corner2 = np.where(grad > 0, 5.0, -5.0)
        fc2 = func(corner2)
        if fc2 < fx_best:
            x_best, fx_best = corner2, fc2

        # Phase 3: Local refinement from best point with adaptive step
        # Use remaining budget for fine-tuning around the best point
        remaining = budget - (1 + 2 * dim + 2)
        if remaining > 0 and time_ok():
            sigma = 0.5
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x_best + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx_best:
                    x_best, fx_best = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if sigma < 1e-6:
                    break

        return x_best, fx_best
# EVOLVE-BLOCK-END
