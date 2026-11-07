"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start Local Optimization for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        self.calls = 0
        budget = self.budget
        dim = self.dim
        
        class StopSearch(Exception):
            pass

        def wrapped_func(x):
            self.calls += 1
            if self.calls > budget:
                raise StopSearch()
            return func(x)

        # Global Phase: Sobol sequence for initial points
        n_global = 64
        sampler = qmc.Sobol(d=dim, scramble=True)
        points = sampler.random(n_global)
        points = points * 10 - 5  # Map [0,1] to [-5,5]
        
        # Evaluate global points
        global_results = []
        for x in points:
            try:
                f_val = wrapped_func(x)
                global_results.append((f_val, x))
            except StopSearch:
                break
        
        if not global_results:
            return None, np.inf

        # Sort by fitness
        global_results.sort(key=lambda item: item[0])
        
        # Local Phase: Refine the best few points
        start_points = [x for _, x in global_results[:5]]
        
        best_x = global_results[0][1]
        best_f = global_results[0][0]
        
        bounds = [(-5, 5)] * dim
        
        for x0 in start_points:
            try:
                # Use L-BFGS-B for bounded local optimization
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except StopSearch:
                # Budget exceeded during local optimization
                break
            except Exception:
                # Catch other errors and continue
                continue
        
        return best_x, best_f
# EVOLVE-BLOCK-END
