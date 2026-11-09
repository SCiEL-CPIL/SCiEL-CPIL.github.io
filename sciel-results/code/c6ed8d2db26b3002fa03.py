"""Global-to-local hybrid optimizer for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Halton-sequence global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    @staticmethod
    def _halton_sequence(n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, 11, ...
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                points[i, d] = r
        return points

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        # Phase 1: Global exploration with Halton sequence
        # Use about 60% of budget for global search
        n_global = min(300, budget // 2)
        halton = self._halton_sequence(n_global, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        points = halton * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls_used = 0

        for i in range(n_global):
            if time.time() > deadline - 50:  # leave time for local phase
                break
            x = points[i]
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement with Nelder-Mead on best point
        if best_x is not None:
            remaining = budget - calls_used
            if remaining > 10:
                # Try Nelder-Mead with tight bounds via penalty
                def penalized(x):
                    # Clip to box and add penalty for out-of-bounds
                    x_clipped = np.clip(x, -5, 5)
                    f_val = func(x_clipped)
                    # Small penalty if any component was clipped
                    penalty = 0.1 * np.sum((x != x_clipped) ** 2)
                    return f_val + penalty

                try:
                    result = minimize(
                        penalized,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 2000),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                        }
                    )
                    if result.fun < best_f:
                        best_x = np.clip(result.x, -5, 5)
                        best_f = func(best_x)
                        calls_used += 1
                except Exception:
                    pass

        # Phase 3: If we still have budget, do a few more local restarts
        # from top few Halton points
        if calls_used < budget and best_x is not None:
            # Find second-best and third-best from Halton evaluation
            # (we didn't store all, so just do a small local search around best)
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
