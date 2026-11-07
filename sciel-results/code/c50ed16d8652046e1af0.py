"""New program: Differential Evolution with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls_made = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal best_x, best_f
            self.calls_made += 1
            if self.calls_made > self.budget:
                return 1e10
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Search with DE
        # Use 80% of budget for DE
        de_budget = int(self.budget * 0.8)
        popsize = 10
        # maxiter such that popsize * (maxiter + 1) <= de_budget
        maxiter = max(1, int(de_budget / popsize) - 1)
        
        try:
            res = differential_evolution(
                counted_func,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                init='sobol',
                tol=1e-8,
                seed=42
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # Phase 2: Local Polish with Nelder-Mead
        # Use remaining budget
        remaining_budget = self.budget - self.calls_made
        if remaining_budget > 10:
            try:
                # maxiter for NM is not directly tied to calls, but we can limit it
                # NM uses ~ dim * maxiter calls? No, it's variable.
                # We just let it run until it converges or budget is hit by counted_func
                res_nm = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
