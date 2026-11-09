"""Rewritten program: deterministic grid-based global search followed by Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = [0]

        def count_call():
            calls[0] += 1

        def wrapped(x):
            count_call()
            return func(x)

        # Phase 1: Coarse grid search over [-4, 4]^dim
        # Use a modest grid to stay within budget
        # For dim=5, we can afford roughly 5-7 points per dimension
        # Let's use 6 points per dimension -> 6^5 = 7776, too many
        # Let's use 5 points per dimension -> 5^5 = 3125, still too many for budget=500
        # So we need to be smarter: use a coarse grid with fewer points
        # With budget=500, we can afford ~400 calls for grid + 100 for NM
        # 4^5 = 1024, still too many
        # 3^5 = 243, that's feasible
        # Let's use 4 points per dimension but only if budget allows
        # Actually, let's use 3 points per dimension for the coarse grid
        grid_points_per_dim = 3
        grid_coords = np.linspace(-4.0, 4.0, grid_points_per_dim)
        
        # Generate all grid points
        grid_axes = np.meshgrid(*([grid_coords] * dim), indexing='ij')
        grid_points = np.array([g.ravel() for g in grid_axes]).T  # shape (N, dim)
        
        n_grid = grid_points.shape[0]
        
        # Evaluate grid points, but cap at budget - 50 to leave room for NM
        max_grid_calls = max(1, budget - 50)
        if n_grid > max_grid_calls:
            # Subsample grid points
            indices = np.linspace(0, n_grid - 1, max_grid_calls, dtype=int)
            grid_points = grid_points[indices]
            n_grid = grid_points.shape[0]
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_grid):
            if calls[0] >= budget:
                break
            x = grid_points[i]
            fx = wrapped(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: single random point
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
        
        # Phase 2: Nelder-Mead refinement from best grid point
        remaining_budget = budget - calls[0]
        if remaining_budget > 5:
            # Use scipy's Nelder-Mead with maxiter controlled by remaining budget
            # Nelder-Mead uses roughly dim+1 evaluations per iteration
            # So maxiter should be roughly remaining_budget // (dim + 1)
            maxiter = max(1, remaining_budget // (dim + 1))
            
            # Create a wrapper that stops when budget is exhausted
            nm_calls = [0]
            
            def nm_func(x):
                nm_calls[0] += 1
                if calls[0] >= budget:
                    return best_f  # Return current best to stop progress
                return func(np.clip(x, -5, 5))
            
            result = minimize(
                nm_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            # Check if NM found something better
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        
        return best_x, best_f
# EVOLVE-BLOCK-END
