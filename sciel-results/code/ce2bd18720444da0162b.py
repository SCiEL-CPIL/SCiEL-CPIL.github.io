import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        max_calls = self.budget
        calls_made = 0
        dim = self.dim

        # Wrapper for counting, clipping, and tracking best
        best_x = None
        best_f = np.inf

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls or time.time() > deadline - 2:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x_clipped, dtype=float)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling to find a good basin
        # Structured points + random points
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
        n_random = 20
        random_pts = rng.uniform(-4, 4, (n_random, dim))
        
        all_pts = np.vstack([structured, random_pts])
        for p in all_pts:
            if calls_made >= max_calls or time.time() > deadline - 5:
                break
            try:
                eval_func(p)
            except Exception:
                break

        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            return np.zeros(dim), 0.0

        # Phase 2: Powell's Method for coarse alignment with Rosenbrock valley
        # Powell aligns search directions with the valley structure, which is 
        # often faster than Nelder-Mead or ES for this specific problem shape.
        remaining = max_calls - calls_made
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': max(1, remaining // 5), 'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': remaining}
                )
                # Powell might have used calls, check if it improved
                # Note: eval_func tracks best_x/best_f internally
            except Exception:
                pass

        # Phase 3: L-BFGS-B Polish
        # L-BFGS-B is excellent for smooth functions once in the basin.
        # It uses gradient approximation and bound constraints.
        remaining = max_calls - calls_made
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    eval_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'maxls': 30, 'gtol': 1e-8, 'ftol': 1e-12}
                )
            except Exception:
                pass

        # Phase 4: Second L-BFGS-B from perturbed point (Multi-start refinement)
        remaining = max_calls - calls_made
        if remaining > 20 and time.time() < deadline - 10:
            try:
                perturb = rng.normal(0, 0.3, dim)
                start2 = np.clip(best_x + perturb, -5, 5)
                res2 = minimize(
                    eval_func,
                    start2,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'maxls': 20, 'gtol': 1e-8, 'ftol': 1e-12}
                )
            except Exception:
                pass

        # Phase 5: Final Nelder-Mead Polish
        # NM can sometimes squeeze out a bit more precision in the final steps.
        remaining = max_calls - calls_made
        if remaining > 30 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    eval_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': max(1, remaining // (dim + 1)), 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
