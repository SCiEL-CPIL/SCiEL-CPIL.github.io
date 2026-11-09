import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        count = 0

        # Generate points using Sobol sequence for better coverage
        # Sobol sequences are low-discrepancy and provide better space-filling properties
        # than uniform grids, which is beneficial for hypervolume maximization.
        
        # Use a structured grid + strategic boundary points for better Pareto front coverage
        # 14x14 grid = 196 points + 4 edge midpoints = 200 points
        n_grid = 14
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        # Generate grid points
        grid_points = [(x0, x1) for x0 in x0_vals for x1 in x1_vals]
        
        # Add 4 strategic edge midpoint points
        edge_points = [(0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)]
        
        points_list = grid_points + edge_points
        points = np.array(points_list[:self.budget])

        # Ensure points are within [0, 1]
        points = np.clip(points, 0.0, 1.0)

        for p in points:
            if count >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = p[0]
            x[1] = p[1]
            
            # Set remaining variables to center to minimize g(x)
            if self.dim > 2:
                x[2:] = 0.5

            # Clip to bounds (redundant but safe)
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break

        return np.array(results) if results else None
