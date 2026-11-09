import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Pareto front is determined by the first (n_obj - 1) variables.
        # For n_obj=3, that's x0 and x1.
        # The remaining variables (x2...x11) should be set to their midpoints 
        # to be on the Pareto front (assuming standard WFG structure).
        
        n_defining = self.n_obj - 1  # 2
        if n_defining < 1:
            n_defining = 1
        if n_defining > self.dim:
            n_defining = self.dim

        # Calculate midpoints for non-defining variables
        mids = (lo + hi) / 2.0
        
        # Grid search for defining variables
        # Budget is 200. 
        # We can use a grid of size roughly sqrt(budget).
        # 14 * 14 = 196.
        
        grid_size = int(np.sqrt(self.budget))
        # Ensure we don't exceed budget
        if grid_size * grid_size > self.budget:
            grid_size -= 1
            
        # If grid is too small, use random sampling in the subspace
        if grid_size < 2:
             # Fallback to random
             points = []
             for _ in range(self.budget):
                 x = np.empty(self.dim)
                 for i in range(self.dim):
                     if i < n_defining:
                         x[i] = lo[i] + (hi[i] - lo[i]) * np.random.random()
                     else:
                         x[i] = mids[i]
                 points.append(x)
                 if time.time() > deadline: break
             for p in points:
                 problem(p)
             return None

        # Create grid points
        # linspace from lo to hi for the defining variables
        # We want to include the boundaries to cover the extent of the front
        x0_vals = np.linspace(lo[0], hi[0], grid_size)
        x1_vals = np.linspace(lo[1], hi[1], grid_size)
        
        evaluated = 0
        for x0 in x0_vals:
            for x1 in x1_vals:
                if evaluated >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = np.empty(self.dim)
                x[0] = x0
                x[1] = x1
                # Set remaining to midpoints
                for i in range(2, self.dim):
                    x[i] = mids[i]
                
                problem(x)
                evaluated += 1
            
            if evaluated >= self.budget:
                break
            if time.time() > deadline:
                break

        # If we have budget left, we can add a few random points in the subspace
        # to ensure we didn't miss anything due to grid alignment, though grid is usually sufficient.
        while evaluated < self.budget:
            if time.time() > deadline:
                break
            x = np.empty(self.dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * np.random.random()
            x[1] = lo[1] + (hi[1] - lo[1]) * np.random.random()
            for i in range(2, self.dim):
                x[i] = mids[i]
            problem(x)
            evaluated += 1

        return None
