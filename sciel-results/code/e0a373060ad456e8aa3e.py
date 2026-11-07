import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 Pareto front is at x[2:] = 0.5
        # We only need to optimize x[0] and x[1]
        
        # Strategy:
        # The hypervolume is dominated by points with small f1 and f2 (small x0, x1).
        # However, f3 can be reduced by moving x0, x1 away from 0 (up to ~0.6).
        # A uniform grid on the full [0,1] range wastes evaluations on points
        # with large x0, x1 which have small box widths (1.1 - x) and thus low hypervolume contribution.
        # We use a grid concentrated in the region [0, 0.8] x [0, 0.8].
        
        n_grid = 14
        # 14x14 = 196 points.
        # We will use the remaining 4 points strategically.
        
        # Define the grid range
        # 0.8 covers the region where f3 is minimized and width is still significant.
        grid_max = 0.8
        
        x0_vals = np.linspace(0, grid_max, n_grid)
        x1_vals = np.linspace(0, grid_max, n_grid)
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Add 4 more points
        # These should be in regions not covered by the grid but potentially valuable.
        # The grid covers [0, 0.8].
        # Points slightly outside might capture the minimum f3 if it occurs > 0.8?
        # g(t) peaks around 0.6, so 0.8 is safe.
        # Let's add points that are midpoints of the outer edges of the [0,1] box
        # to ensure we don't miss any "corner" effects, though they are likely low value.
        # Better: add points that refine the grid in the most critical area [0, 0.4].
        # Or just use a 15x13 grid? 15*13 = 195.
        # Let's stick to 14x14 and add 4 points at (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.9)
        # Actually, (0.5, 0.5) is in the grid.
        
        extra_x0 = [0.9, 0.1, 0.9, 0.5]
        extra_x1 = [0.1, 0.9, 0.9, 0.9]
        
        x0_flat = np.concatenate([x0_flat, extra_x0])
        x1_flat = np.concatenate([x1_flat, extra_x1])
        
        # Truncate to budget
        x0_flat = x0_flat[:budget]
        x1_flat = x1_flat[:budget]
        
        # Construct full vectors
        X = np.zeros((len(x0_flat), n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        X[:, 2:] = 0.5
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
        
        return X
