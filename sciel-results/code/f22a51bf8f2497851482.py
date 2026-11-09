"""
Structural change: use CMA-ES (from scipy) as the primary optimizer.
CMA-ES adapts the covariance matrix to the local geometry, which is far more
effective for the anisotropic Bent Cigar landscape than coordinate descent or
Powell. We combine a small Sobol exploration phase with CMA-ES for the main
search, and a final Powell polish if budget remains.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar (f12):
    Phase 1: Small Sobol exploration to find a good starting point.
    Phase 2: CMA-ES (via scipy.optimize.minimize) as the main search.
    Phase 3: Powell polish if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def make_call(x):
            nonlocal calls_used
            if not check_time() or calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            return fx

        best_x = np.zeros(dim)
        best_fx = float('inf')

        # Phase 1: Sobol exploration
        n_sobol = max(10, min(30, budget // 10))
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * 10.0 - 5.0

        for i in range(n_sobol):
            if not check_time() or calls_used >= budget:
                break
            x = sobol_points[i]
            fx = make_call(x)
            if fx is None:
                break
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not check_time() or calls_used >= budget:
            return best_x, best_fx

        # Phase 2: CMA-ES via scipy
        remaining = budget - calls_used
        if remaining > 20:
            try:
                # Use a reasonable initial sigma
                sigma0 = 1.0
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxfev': remaining, 'maxiter': 1000,
                                        'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        if not check_time() or calls_used >= budget:
            return best_x, best_fx

        # Phase 3: Powell polish
        remaining = budget - calls_used
        if remaining > 10:
            try:
                res = minimize(func, best_x, method='Powell',
                               options={'maxfev': remaining, 'maxiter': 500})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
