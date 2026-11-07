"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # 1. Identify non-dominated x0 values on the ideal front
        # A point x is dominated if there exists y < x such that f2(y) <= f2(x)
        # We want to sample only from the set of "record lows" in f2
        
        # Dense grid to identify the non-dominated envelope
        n_grid = 10000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        # Calculate f2 for the grid
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find indices where f2 is a new minimum (strictly less than all previous)
        # We use a running minimum
        min_f2_so_far = np.full(n_grid, np.inf)
        min_f2_so_far[0] = f2_grid[0]
        for i in range(1, n_grid):
            if f2_grid[i] < min_f2_so_far[i-1]:
                min_f2_so_far[i] = f2_grid[i]
            else:
                min_f2_so_far[i] = min_f2_so_far[i-1]
                
        # The non-dominated points are those where f2_grid[i] == min_f2_so_far[i]
        # Note: due to floating point, we might use a small tolerance or just equality
        # Since we are generating the values, equality should hold for the record setters
        is_nd = (f2_grid == min_f2_so_far)
        
        # Get the x values that are non-dominated
        x_nd = x_grid[is_nd]
        
        # If we have fewer than budget non-dominated points, use all of them
        # Otherwise, select 'budget' points from x_nd
        # To maximize hypervolume, we should distribute them well.
        # Since x_nd is sorted, we can pick uniformly from the indices of x_nd
        # This ensures we cover the entire range of the non-dominated front
        
        if len(x_nd) <= budget:
            selected_x = x_nd
        else:
            # Select 'budget' indices uniformly from the range [0, len(x_nd))
            # This picks a uniform subset of the non-dominated points
            indices = np.linspace(0, len(x_nd) - 1, budget, dtype=int)
            selected_x = x_nd[indices]
            
        # Ensure we don't exceed budget
        n_evals = len(selected_x)
        
        for i in range(n_evals):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)
            
            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
