import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with Rastrigin-aware initialization for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        def time_left():
            return deadline - time.time()

        def do_eval(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Evaluate origin first
        do_eval(np.zeros(dim))

        # Generate diverse initial points
        # 1. Random points in the box
        n_random = 30
        for _ in range(n_random):
            if eval_count >= budget or time_left() < 5.0:
                break
            x = rng.uniform(-5, 5, size=dim)
            do_eval(x)

        # 2. Points near integer grid (Rastrigin has local minima near integers)
        n_grid = 20
        for _ in range(n_grid):
            if eval_count >= budget or time_left() < 5.0:
                break
            x = np.round(rng.uniform(-4, 4, size=dim))
            x = np.clip(x, -5, 5)
            do_eval(x)

        # 3. Points with small perturbations around best found so far
        n_perturb = 15
        for _ in range(n_perturb):
            if eval_count >= budget or time_left() < 5.0:
                break
            x = best_x + rng.normal(0, 0.5, size=dim)
            x = np.clip(x, -5, 5)
            do_eval(x)

        # Now run L-BFGS-B from multiple starting points
        # Pick the best 10 starting points from our evaluations
        # We need to track which points we've evaluated
        # Let's just restart L-BFGS-B from the current best and a few random ones

        n_restarts = 12
        for r in range(n_restarts):
            if eval_count >= budget or time_left() < 10.0:
                break

            if r == 0:
                x0 = best_x.copy()
            else:
                # Mix of random and best-based starts
                if r % 2 == 0:
                    x0 = best_x + rng.normal(0, 1.0, size=dim)
                else:
                    x0 = rng.uniform(-5, 5, size=dim)
                x0 = np.clip(x0, -5, 5)

            # Limit function calls per L-BFGS-B run
            maxfun = max(10, (budget - eval_count) // (n_restarts - r))
            maxiter = max(20, maxfun // 2)

            try:
                result = minimize(
                    do_eval,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxfun': maxfun, 'maxiter': maxiter, 'gtol': 1e-8}
                )
            except Exception:
                continue

            if eval_count >= budget or time_left() < 5.0:
                break

            # After L-BFGS-B, do a few local perturbations
            if time_left() > 5.0:
                for _ in range(5):
                    if eval_count >= budget or time_left() < 3.0:
                        break
                    x = best_x + rng.normal(0, 0.1, size=dim)
                    x = np.clip(x, -5, 5)
                    do_eval(x)

        # Final polish with very small steps
        if time_left() > 3.0:
            for _ in range(20):
                if eval_count >= budget or time_left() < 2.0:
                    break
                x = best_x + rng.normal(0, 0.01, size=dim)
                x = np.clip(x, -5, 5)
                do_eval(x)

        return best_x, best_f
