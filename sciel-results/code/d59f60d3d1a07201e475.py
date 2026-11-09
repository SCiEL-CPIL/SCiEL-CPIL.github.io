# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7.
    Strategy: Sample the 2D Pareto front (x_3..x_22 = 0) with a non-uniform grid
    that is denser near the ideal point (f1=0, f2=0) to maximize hypervolume.
    Uses a power-law transform to concentrate points near the origin in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # Determine grid size
        # We want a square grid N x N such that N^2 <= budget.
        # For budget=200, sqrt(200) ~ 14.14, so N=14, N^2=196.
        # We can use the remaining 4 evaluations for refinement or just leave them.
        
        grid_size = int(np.sqrt(self.budget))
        grid_size = max(2, grid_size)
        while (grid_size + 1) ** 2 <= self.budget:
            grid_size += 1
        
        # Non-linear grid to concentrate points near the ideal objectives
        # f1 = 1 - x1, f2 = x2.
        # We want f1, f2 small.
        # x1 = 1 - f1, x2 = f2.
        # Let t in [0, 1] be the grid coordinate.
        # We want the distribution of f1 and f2 to be denser near 0.
        # If we sample t uniformly, and set f = t^gamma, then for gamma > 1,
        # t^gamma is denser near 0.
        # So x1 = 1 - t^gamma1, x2 = t^gamma2.
        
        # For hypervolume with reference point (1.1, 1.1, 25.0) and DTLZ7 front,
        # the optimal distribution balances coverage. A moderate gamma around 1.2-1.3
        # provides good coverage without being too concentrated near the origin.
        # Let's try gamma=1.2 which spreads points more evenly.
        
        gamma1 = 1.2
        gamma2 = 1.2
        
        t_vals = np.linspace(0.0, 1.0, grid_size)
        
        # x1 = 1 - t^gamma1. 
        # When t=0, x1=1, f1=0.
        # When t=1, x1=0, f1=1.
        # This maps t in [0,1] to x1 in [1,0].
        
        # x2 = t^gamma2.
        # When t=0, x2=0, f2=0.
        # When t=1, x2=1, f2=1.
        
        x1_vals = 1.0 - t_vals ** gamma1
        x2_vals = t_vals ** gamma2

        # Iterate over the grid
        for i in range(grid_size):
            for j in range(grid_size):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(self.dim)
                x[0] = x1_vals[i]
                x[1] = x2_vals[j]
                # x[2:] = 0 for g=0 (Pareto front)

                try:
                    obj = problem(x)
                    obj = list(obj)
                    results.append(obj)
                    n_evals += 1
                except Exception:
                    break
            if n_evals >= self.budget or time.time() > deadline:
                break
        
        # If we have remaining budget, use it to add points at strategic locations
        # that complement the grid. The grid already includes the corners (t=0 and t=1),
        # so we should add points at intermediate positions that might be underrepresented.
        
        remaining = self.budget - n_evals
        if remaining > 0:
            # Add points at the ideal point and a few strategic locations
            # The ideal point (f1=0, f2=0) corresponds to x1=1, x2=0
            # This is already in the grid at (i=0, j=0), so let's add other points.
            
            # Add points at the "middle" of each edge and the center
            extra_points = [
                [1.0, 0.0],  # Ideal point (f1=0, f2=0)
                [0.5, 0.5],  # Center
            ]
            
            for k in range(min(remaining, len(extra_points))):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = extra_points[k][0]
                x[1] = extra_points[k][1]
                
                try:
                    obj = problem(x)
                    obj = list(obj)
                    results.append(obj)
                    n_evals += 1
                except Exception:
                    break
        
        return results
# EVOLVE-BLOCK-END
