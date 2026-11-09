import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        x_best = np.zeros(self.dim)
        f_best = float('inf')
        calls_used = 0

        # Phase 1: Halton Exploration
        n_halton = min(60, self.budget)
        halton_pts_01 = self._halton_sequence(n_halton, self.dim)
        halton_pts = -5.0 + 10.0 * halton_pts_01

        values = np.full(n_halton, np.inf)
        for i in range(n_halton):
            if calls_used >= self.budget or time.time() > deadline:
                break
            val = func(halton_pts[i])
            values[i] = val
            calls_used += 1
            if val < f_best:
                f_best = val
                x_best = halton_pts[i].copy()

        # Phase 2: Nelder-Mead Refinement
        sorted_indices = np.argsort(values)
        n_refine = 3
        remaining = self.budget - calls_used
        
        # Allocate budget equally among top candidates
        budget_per = remaining // n_refine if n_refine > 0 else 0
        
        for rank in range(n_refine):
            if calls_used >= self.budget or time.time() > deadline:
                break
            
            idx = sorted_indices[rank]
            x0 = halton_pts[idx].copy()
            
            # Nelder-Mead maxiter: 1 iteration involves up to dim+1 function evaluations.
            # We estimate maxiter to stay within budget_per.
            max_iter = max(20, budget_per // (self.dim + 1))
            
            def wrapped_func(x):
                nonlocal calls_used, f_best, x_best
                if calls_used >= self.budget or time.time() > deadline:
                    raise RuntimeError("Budget or time exceeded")
                val = func(x)
                calls_used += 1
                if val < f_best:
                    f_best = val
                    x_best = x.copy()
                return val

            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x.copy()
            except Exception:
                pass

        return x_best, f_best
