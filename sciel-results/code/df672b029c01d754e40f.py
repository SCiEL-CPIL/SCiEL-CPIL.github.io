"""Rewritten program: deterministic grid-based global search followed by multi-start L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based global search + Multi-start L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            return func(x)

        # Phase 1: Coarse grid search over [-4, 4]^dim
        grid_points_per_dim = 3
        grid_coords = np.linspace(-4.0, 4.0, grid_points_per_dim)
        
        grid_axes = np.meshgrid(*([grid_coords] * dim), indexing='ij')
        grid_points = np.array([g.ravel() for g in grid_axes]).T
        n_grid = grid_points.shape[0]
        
        # Store all grid results to pick top K
        grid_results = []
        for i in range(n_grid):
            if calls[0] >= budget:
                break
            x = grid_points[i]
            fx = wrapped(x)
            grid_results.append((fx, x))
            
        if not grid_results:
            return np.zeros(dim), np.inf

        # Sort by function value
        grid_results.sort(key=lambda t: t[0])
        
        # Take top 3 distinct points
        top_points = []
        for fx, x in grid_results:
            if len(top_points) >= 3:
                break
            # Ensure points are not identical (grid points are unique, so this is safe)
            top_points.append(x)
            
        best_f = grid_results[0][0]
        best_x = grid_results[0][1].copy()
        
        # Phase 2: Multi-start L-BFGS-B
        remaining_budget = budget - calls[0]
        n_starts = len(top_points)
        
        if remaining_budget > 10 and n_starts > 0:
            # Split budget among starts
            budget_per_start = remaining_budget // n_starts
            
            for x0 in top_points:
                if calls[0] >= budget:
                    break
                
                # If we don't have enough budget for a meaningful run, skip
                if budget_per_start < 10:
                    continue
                    
                maxiter = max(1, budget_per_start // (2 * dim))
                
                def lbf_func(x):
                    if calls[0] >= budget:
                        return best_f
                    return func(np.clip(x, -5, 5))
                
                bounds = [(-5.0, 5.0)] * dim
                
                try:
                    result = minimize(
                        lbf_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': maxiter,
                            'ftol': 1e-12,
                            'gtol': 1e-8,
                            'maxls': 20
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
