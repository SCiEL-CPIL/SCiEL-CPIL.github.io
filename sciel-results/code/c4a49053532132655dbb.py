"""Rewritten optimizer: multi-start L-BFGS-B with finite-difference gradients."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with a deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # 300s limit, 30s safety margin
        rng = np.random.default_rng(42)

        # Wrap func to count calls and enforce budget
        call_count = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            if call_count[0] >= budget:
                return best_f if np.isfinite(best_f) else 0.0
            call_count[0] += 1
            x_clip = np.clip(x, -5.0, 5.0)
            f = float(func(x_clip))
            if f < best_f:
                best_f = f
                best_x = x_clip.copy()
            return f

        # Initial points: center, random, and a few structured starts
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, dim))
        # Add a point near the typical Rosenbrock valley
        starts.append(np.ones(dim) * 0.5)

        for start in starts:
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            try:
                res = minimize(
                    wrapped,
                    start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': 80, 'ftol': 1e-12, 'gtol': 1e-8},
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # If we still have budget, do a local (1+1)-ES polish
        if best_x is not None and time.time() < deadline and call_count[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while call_count[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = wrapped(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
