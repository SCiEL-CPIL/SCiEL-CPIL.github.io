"""Combined optimizer: LHS global scan -> Powell -> CMA-ES -> Nelder-Mead -> precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase black-box minimizer with explicit budget and deadline guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before the 330s kill
        dim = self.dim
        budget = self.budget

        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if time.time() > deadline or calls[0] >= budget:
                return float('inf')
            calls[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.asarray(x, dtype=np.float64).copy()
            return val

        # ---- Phase 1: Latin Hypercube global scan -----------------------------
        # Spread a handful of points across the box to locate the shifted basin
        # quickly (early progress matters for AOCC).
        n_lhs = min(15, max(1, budget - 5))
        if n_lhs > 0 and time.time() < deadline:
            lhs = np.empty((n_lhs, dim), dtype=np.float64)
            for i in range(n_lhs):
                for d in range(dim):
                    lo = -5.0 + 10.0 * i / n_lhs
                    hi = -5.0 + 10.0 * (i + 1) / n_lhs
                    lhs[i, d] = np.random.uniform(lo, hi)
            for x in lhs:
                if calls[0] >= budget or time.time() > deadline:
                    break
                wrapped_func(x)

        if best_x is None:  # no successful call; fall back to the origin
            best_x = np.zeros(dim, dtype=np.float64)
            best_f = wrapped_func(best_x)

        bounds = [(-5.0, 5.0)] * dim

        # ---- Phase 2: Powell (derivative-free, coordinate-wise) ---------------
        if calls[0] < budget and time.time() < deadline:
            try:
                res = minimize(wrapped_func, best_x, method='Powell',
                               bounds=bounds,
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-14})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.asarray(res.x, dtype=np.float64).copy()
            except Exception:
                pass

        # ---- Phase 3: CMA-ES (adaptive covariance, robust to rotation) --------
        # Bridges Powell's axis-aligned moves with the final local polish.
        if calls[0] < budget and time.time() < deadline:
            try:
                # Pick a modest sigma: small if Powell converged tightly, else medium.
                span = float(np.max(np.ptp(best_x)) if dim else 0.0)
                sigma0 = max(0.05, min(0.5, 0.05 * (1.0 + span)))
                res = minimize(wrapped_func, best_x, method='CMA',
                               options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-13,
                                        'popsize': 8, 'sigma0': sigma0})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.asarray(res.x, dtype=np.float64).copy()
            except Exception:
                pass

        # ---- Phase 4: Nelder-Mead (final local convergence) -------------------
        if calls[0] < budget and time.time() < deadline:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-15,
                                        'initial_simplex': 0.05})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.asarray(res.x, dtype=np.float64).copy()
            except Exception:
                pass

        # ---- Phase 5: Precision polish (tiny random nudges + tight NM) --------
        if calls[0] < budget - 5 and time.time() < deadline:
            try:
                for _ in range(min(4, (budget - calls[0]) // 2)):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    cand = np.clip(best_x + np.random.normal(0.0, 1e-4, dim), -5.0, 5.0)
                    wrapped_func(cand)
                if calls[0] < budget - 3 and time.time() < deadline:
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': 40, 'xatol': 1e-12, 'fatol': 1e-16,
                                            'initial_simplex': 1e-4})
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.asarray(res.x, dtype=np.float64).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
