"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


class Optimizer:
    """Grid-based global exploration followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Coarse Grid Exploration
        # Use 3 levels per dimension to cover the domain [-5, 5]^dim
        # This creates a 3^dim grid. For dim=5, that's 243 points.
        # This leaves budget - 243 for local search.
        
        levels = np.linspace(-5, 5, 3)
        grid_points = np.array(np.meshgrid(*([levels] * self.dim))).T.reshape(-1, self.dim)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate grid points
        # To improve early AOCC, we might want to find a good point quickly.
        # Evaluating the center (0,0,...,0) first is a good heuristic for BBOB functions
        # which are often centered or have a global minimum near the origin in the transformed space.
        # However, the grid points are fixed. 
        # Let's just iterate through the grid.
        
        for x in grid_points:
            if calls_made >= self.budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local Refinement
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 0 and best_x is not None:
            # Use Nelder-Mead for local optimization
            # We need to ensure we don't exceed the budget.
            # scipy.optimize.minimize does not have a strict max_func_calls option for Nelder-Mead
            # that stops exactly when the limit is reached in all versions/behaviors.
            # We wrap the function to count calls and return a penalty or stop.
            
            calls_left = remaining_budget
            
            def wrapped_func(x):
                nonlocal calls_left
                if calls_left <= 0:
                    return float('inf')
                calls_left -= 1
                return func(x)

            try:
                # Nelder-Mead is suitable for small dimensions and few evaluations
                res = so.minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                  options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass # If optimization fails, we keep the best grid point

        return best_x, best_f
# EVOLVE-BLOCK-END
