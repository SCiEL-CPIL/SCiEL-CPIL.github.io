import numpy as np
import time
from scipy.optimize import minimize


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
        n_random = 40
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

        # Phase 2: L-BFGS-B from best point
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

        # Phase 3: If budget remains, try a second L-BFGS-B from a slightly perturbed point
        remaining = self.budget - call_count
        if remaining > 20 and time.time() < deadline - 10:
            try:
                perturb = rng.normal(0, 0.3, self.dim)
                start2 = np.clip(best_x + perturb, -5, 5)
                res2 = minimize(
                    safe_func,
                    start2,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'maxls': 20, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = np.array(res2.x, dtype=float)
            except Exception:
                pass

        # Phase 4: Final polish with Nelder-Mead if budget remains
        remaining = self.budget - call_count
        if remaining > 30 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
