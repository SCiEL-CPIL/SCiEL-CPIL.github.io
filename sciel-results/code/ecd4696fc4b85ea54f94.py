import numpy as np
import time
from scipy.optimize import minimize
import math


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Initial sampling to find the basin
        # Include origin, structured points, and random samples
        n_structured = 8
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim),
            np.ones(self.dim) * -1.0,
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 0.1,
        ]
        n_random = 30
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))

        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 5:
                break
            try:
                safe_func(p)
            except ValueError:
                break

        if best_x is None:
            # Fallback: return a default point
            return np.zeros(self.dim), 0.0

        # Phase 2: L-BFGS-B from best point with tighter tolerances
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 300, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Phase 3: If budget remains, try Powell's method for valley alignment
        remaining = self.budget - call_count
        if remaining > 30 and time.time() < deadline - 10:
            try:
                res_powell = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'xtol': 1e-10, 'ftol': 1e-12}
                )
                if res_powell.fun < best_f:
                    best_f = float(res_powell.fun)
                    best_x = np.array(res_powell.x, dtype=float)
            except Exception:
                pass

        # Phase 4: Final high-precision polish with L-BFGS-B
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 5:
            try:
                res_polish = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'maxls': 40, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                if res_polish.fun < best_f:
                    best_f = float(res_polish.fun)
                    best_x = np.array(res_polish.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
