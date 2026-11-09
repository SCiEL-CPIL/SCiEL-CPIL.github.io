"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0

        # Sobol sequence for systematic sampling
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = min(20, budget)
        # Generate Sobol points in [0,1]^dim
        sobol_points = sampler.random(n_sobol)
        # Map to [-5, 5]^dim
        sobol_points = sobol_points * (upper - lower) + lower

        best_x = None
        best_f = np.inf

        for i in range(n_sobol):
            x = sobol_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_made = n_sobol
        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_f

        # Nelder-Mead refinement starting from the best Sobol point
        # We need to be careful about the number of function evaluations.
        # Nelder-Mead in scipy uses maxiter and maxfev.
        # Let's set maxfev to remaining budget to not exceed it.
        
        # Note: scipy's minimize with Nelder-Mead does not strictly respect a hard stop on 
        # total function calls including the initial ones, but we can pass maxfev.
        # However, we've already made n_sobol calls. So maxfev for the optimizer should be remaining.
        
        # Also, we should ensure the initial simplex is within bounds or handle bounds.
        # Nelder-Mead doesn't support bounds directly in scipy. 
        # We can use a penalty or just rely on the fact that the function clips.
        # The task says func(x) clips a finite length-5 vector to the box.
        # So we don't need to worry about bounds in the optimizer; the function handles it.
        
        # Initialize simplex
        # The default simplex in scipy is constructed from the initial point.
        # We can let scipy handle it, or provide a custom one.
        # Let's let scipy handle it for simplicity, but ensure maxfev is set.
        
        try:
            result = minimize(
                func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'maxfev': remaining, 'adaptive': True}
            )
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception:
            # If Nelder-Mead fails for any reason, we just keep the best Sobol point
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
