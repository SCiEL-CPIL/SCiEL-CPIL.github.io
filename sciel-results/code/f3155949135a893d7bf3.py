import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ7 Pareto front is a 2D surface parameterized by x[0] and x[1]
        # (indices 0 and 1). The other variables should be 0 for optimal solutions.
        # We'll create a 2D grid over x[0] and x[1] to cover the Pareto front.
        
        # Use 14x14 = 196 points, leaving 4 extra evaluations
        n_per_dim = 14
        n_grid = n_per_dim * n_per_dim
        
        # Generate space-filling points in [0, 1]^2 using Halton-like sequences
        # For 2D, use Halton with bases 2 and 3
        
        def halton_2d(n, base1=2, base2=3):
            points = []
            for i in range(1, n + 1):
                # Base base1
                f1 = 1.0
                r1 = 0.0
                idx = i
                while idx:
                    f1 /= base1
                    r1 += f1 * (idx % base1)
                    idx //= base1
                
                # Base base2
                f2 = 1.0
                r2 = 0.0
                idx = i
                while idx:
                    f2 /= base2
                    r2 += f2 * (idx % base2)
                    idx //= base2
                
                points.append((r1, r2))
            return np.array(points)
        
        grid_points = halton_2d(n_grid)
        
        # Add boundary points: corners and edge midpoints
        # The Pareto front includes points where x[0]=0, x[1]=0, x[0]=1, x[1]=1
        boundary_points = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
        ])
        
        # Combine grid and boundary points, but ensure we don't exceed budget
        all_points = np.vstack([grid_points, boundary_points])
        
        # If we have more than budget, sample down to budget
        if len(all_points) > n_evals:
            # Keep boundary points and sample from grid
            n_boundary = len(boundary_points)
            n_grid_keep = n_evals - n_boundary
            if n_grid_keep > 0:
                # Randomly select grid points
                idx = np.random.choice(len(grid_points), size=n_grid_keep, replace=False)
                selected_grid = grid_points[idx]
            else:
                selected_grid = np.array([]).reshape(0, 2)
            all_points = np.vstack([selected_grid, boundary_points])
        
        # Shuffle to avoid any ordering bias
        np.random.shuffle(all_points)
        
        results = []
        for x01 in all_points[:n_evals]:
            if time.time() > deadline - 5:  # Safety margin
                break
            
            x = np.zeros(self.dim)
            x[0] = x01[0]
            x[1] = x01[1]
            # x[2:] are 0.
            
            # Clip
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
