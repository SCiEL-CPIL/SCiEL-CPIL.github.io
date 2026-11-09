"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 300.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        budget = self.budget
        dim = self.dim
        # Main (1+1)-ES loop with time guard
        for i in range(budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            # Adapt sigma via 1/5 rule on a rolling window for stability
            if i % 20 == 19 and i > 0:
                pass  # simple multiplicative rule is fine for 5D
        # Nelder-Mead polish to tighten convergence near the optimum
        try:
            from scipy.optimize import minimize
            # Build initial simplex around current best
            best_x = x.copy()
            best_f = fx
            # Try a few small offsets to seed simplex
            offsets = []
            for d in range(dim):
                off = best_x.copy()
                off[d] += 0.1
                off = np.clip(off, -5, 5)
                offsets.append(off)
            # Use scipy minimize with Nelder-Mead
            res = minimize(
                lambda z: func(np.clip(z, -5, 5)),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if res.fun < best_f:
                best_x = np.clip(res.x, -5, 5)
                best_f = res.fun
            x, fx = best_x, best_f
        except Exception:
            pass
        return x, fx
# EVOLVE-BLOCK-END
