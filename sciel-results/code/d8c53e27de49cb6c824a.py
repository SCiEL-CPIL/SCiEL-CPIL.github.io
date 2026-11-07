import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        calls = 0
        dim = self.dim
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                raise StopIteration
            if time.time() > deadline:
                raise StopIteration
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=np.float64)
            return val

        # Phase 1: Initial sampling
        rng = np.random.default_rng(42)
        structured = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
            np.ones(dim),
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 0.1,
        ]
        n_random = 15
        random_pts = rng.uniform(-4, 4, (n_random, dim))

        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if calls >= self.budget or time.time() > deadline - 5:
                break
            try:
                safe_func(p)
            except StopIteration:
                break

        if best_x is None:
            return np.zeros(dim), 0.0

        # Phase 2: (1+1)-ES with aggressive sigma adaptation
        # Use 50% of remaining budget for ES
        remaining = self.budget - calls
        es_budget = int(remaining * 0.5)
        es_budget = max(es_budget, 30)

        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        success_hist = []
        hist_len = 12

        for _ in range(es_budget):
            if time.time() > deadline - 10 or calls >= self.budget:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            try:
                fy = safe_func(y)
            except StopIteration:
                break

            if fy < fx:
                x, fx = y, fy
                success_hist.append(1)
            else:
                success_hist.append(0)

            if len(success_hist) > hist_len:
                success_hist.pop(0)

            if len(success_hist) == hist_len:
                rate = np.mean(success_hist)
                if rate > 0.35:
                    sigma *= 1.12
                elif rate < 0.15:
                    sigma *= 0.88

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # Phase 3: L-BFGS-B polish from best point
        remaining = self.budget - calls
        if remaining > 20 and time.time() < deadline - 15:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'maxls': 30,
                             'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=np.float64)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 4: Second L-BFGS-B from perturbed point if budget remains
        remaining = self.budget - calls
        if remaining > 30 and time.time() < deadline - 10:
            try:
                perturb = rng.normal(0, 0.3, dim)
                start2 = np.clip(best_x + perturb, -5.0, 5.0)
                res2 = minimize(
                    safe_func,
                    start2,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'maxls': 20,
                             'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res2.fun < best_f:
                    best_f = float(res2.fun)
                    best_x = np.array(res2.x, dtype=np.float64)
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 5: Final Nelder-Mead polish if budget remains
        remaining = self.budget - calls
        if remaining > 40 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=np.float64)
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_f
