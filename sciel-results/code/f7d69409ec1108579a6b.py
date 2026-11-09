"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Sobol global search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0
        
        # Phase 1: Global exploration with Sobol sequence
        # Use ~20% of budget for global search to ensure early progress
        global_budget = max(10, int(budget * 0.2))
        
        # Generate Sobol points
        # scipy.stats.qmc.Sobol requires dimension
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Generate slightly more points than needed to account for clipping or if we want to be safe
        # Sobol requires power of 2 for best properties, but we can just take the first N
        sobol_points = sampler.random(global_budget)
        # Map [0,1] to [lb, ub]
        sobol_points = sobol_points * (ub - lb) + lb
        
        x_best = None
        f_best = np.inf
        
        # Evaluate global points
        for i in range(global_budget):
            x = sobol_points[i]
            # Ensure within bounds (Sobol is [0,1], so it is, but clip just in case of floating point)
            x = np.clip(x, lb, ub)
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        
        # Phase 2: Local refinement with Nelder-Mead
        # Use remaining budget
        local_budget = budget - global_budget
        
        if local_budget > 0:
            # Nelder-Mead parameters
            # maxiter is not the number of function calls, but iterations.
            # In 5D, each NM iteration uses ~6 function evaluations (simplex update).
            # So maxiter should be roughly local_budget / (dim + 1)
            max_iter = max(1, local_budget // (dim + 1))
            
            # Define the function for scipy
            def nm_func(x):
                x_clipped = np.clip(x, lb, ub)
                return func(x_clipped)
            
            # Run Nelder-Mead
            try:
                res = minimize(nm_func, x_best, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                # Update best if NM found something better
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x
            except Exception:
                pass
        
        return x_best, f_best
# EVOLVE-BLOCK-END
