"""Rewritten optimizer for BBOB f7 (Step ellipsoidal).
Uses CMA-ES (from scipy) with a time guard, plus a fallback (1+1)-ES if CMA-ES
fails to converge quickly.  Multi-start initialization gives a head start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with fallback and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        max_calls = self.budget  # 500

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        def track_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # --- Phase 1: Multi-start initialization to find a good starting point ---
        n_starts = 10
        best_x = None
        best_f = np.inf

        for i in range(n_starts):
            if not elapsed_ok():
                break
            x = rng.uniform(-4.5, 4.5, self.dim)
            fx = track_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = track_call(best_x)

        # --- Phase 2: Run CMA-ES from the best starting point ---
        # We wrap func to count calls and enforce the budget
        cma_calls = [0]

        def cma_func(x):
            nonlocal calls_made
            if calls_made >= max_calls or time.time() >= deadline:
                return np.inf
            cma_calls[0] += 1
            calls_made += 1
            return func(x)

        # CMA-ES parameters
        x0 = best_x.copy()
        sigma0 = 1.5

        try:
            result = minimize(
                cma_func,
                x0,
                method='CMA',
                options={
                    'maxiter': 200,
                    'tol': 1e-10,
                    'verbose': 0,
                    'bounds': [(-5, 5)] * self.dim,
                }
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except Exception:
            pass

        # --- Phase 3: If CMA-ES used few calls, do a local (1+1)-ES refinement ---
        if elapsed_ok() and calls_made < max_calls:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            min_sigma = 0.001
            max_sigma = 5.0
            lambda_ = 6
            stagnation = 0
            stagnation_limit = 30

            while elapsed_ok():
                improved = False
                for j in range(lambda_):
                    if not elapsed_ok():
                        break
                    step = rng.standard_normal(self.dim) * sigma
                    y = np.clip(x + step, -5, 5)
                    fy = track_call(y)

                    if fy < fx:
                        x = y
                        fx = fy
                        improved = True
                        sigma = min(sigma * 1.2, max_sigma)
                        stagnation = 0
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                    else:
                        sigma = max(sigma * 0.85, min_sigma)

                if not improved:
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation >= stagnation_limit:
                    # Try a random restart
                    if not elapsed_ok():
                        break
                    for _ in range(3):
                        if not elapsed_ok():
                            break
                        y = rng.uniform(-4.5, 4.5, self.dim)
                        fy = track_call(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma = 1.5
                            stagnation = 0
                        elif fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                            stagnation = 0
                    stagnation = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
