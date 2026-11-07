"""Multi-start Nelder-Mead with time guard for Katsuura (f23) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead exploiting Katsuura's multimodality."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            eval_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate diverse starting points
        n_starts = 20
        # Mix of random points and structured points
        starts = []
        # Random uniform
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, self.dim))
        # Structured: corners and midpoints of sub-boxes
        for _ in range(5):
            starts.append(rng.uniform(-3, 3, self.dim))
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, self.dim) * 0.5)

        # Shuffle starts for diversity
        idx = rng.permutation(len(starts))
        starts = [starts[i] for i in idx]

        # Run Nelder-Mead from each start, tracking best
        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break

            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(self.budget - eval_count, 150),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                continue

            # If we've used most of our budget, stop
            if eval_count >= self.budget - 10:
                break

        # If we still have budget left, do a final fine-tuning around the best
        if best_x is not None and eval_count < self.budget - 5 and time.time() < deadline:
            try:
                res2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': self.budget - eval_count,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = np.clip(res2.x, -5.0, 5.0)
            except Exception:
                pass

        if best_x is None:
            # Fallback: return first evaluation
            x = rng.uniform(-5, 5, self.dim)
            best_x = x
            best_f = func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
