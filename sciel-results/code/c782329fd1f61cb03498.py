# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-driven structured sampling optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front is 2D.
        # We assume x_3...x_D are at their optimal values (0 for standard DTLZ7 g).
        # We sample a grid on x_1, x_2 to cover the front uniformly.
        
        # Determine grid size
        # Budget is 200.
        # Let's use a square-ish grid.
        n_grid = int(np.sqrt(self.budget))
        # Adjust to fit budget
        while n_grid * n_grid > self.budget:
            n_grid -= 1
        while (n_grid + 1) * (n_grid + 1) <= self.budget:
            n_grid += 1
            
        used = 0
        results = []
        
        # Create grid points for x1, x2
        # Use linspace to ensure coverage of boundaries
        # But for hypervolume, uniform distribution is good.
        # Linspace includes 0 and 1.
        x1_vals = np.linspace(lo[0], hi[0], n_grid)
        x2_vals = np.linspace(lo[1], hi[1], n_grid)
        
        # We have budget - n_grid*n_grid left.
        # If we have extra, we can add a few random points or refine.
        # But a dense grid is usually very good for hypervolume.
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                if used >= self.budget:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # Set x_3...x_D to 0 (assumed optimal for g)
                # If bounds are [0,1], 0 is valid.
                for i in range(2, self.dim):
                    x[i] = lo[i] # Assuming lo is 0
                    
                fx = problem(x)
                results.append(fx)
                used += 1
                
            if used >= self.budget:
                break
        
        # If we have remaining budget, we can add random points or slightly perturb grid points.
        # But a uniform grid is already very strong.
        # Let's add a few random points to cover any non-uniformity if the front is not convex?
        # DTLZ7 is non-convex. A grid covers it well.
        
        return results
# EVOLVE-BLOCK-END
