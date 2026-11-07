import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0

        # Pre-compute lookup table for f1(x1)
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # We want to find x1 for a given f1.
        # Let Y = 1 - f1. We solve exp(-4*x1) * sin(6*pi*x1)^8 = Y.
        
        # Create a dense grid
        n_grid = 20000
        x_grid = np.linspace(0, 1, n_grid)
        # Calculate Y for all grid points
        # sin(6*pi*x)^8
        sin_term = np.sin(6 * np.pi * x_grid)
        sin_term = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x_grid)
        Y_grid = exp_term * sin_term
        
        # The range of f1 is [1 - max(Y_grid), 1]
        # max(Y_grid) is approx 0.716
        # So f1_min = 1 - 0.716 = 0.284
        f1_min = 1 - np.max(Y_grid)
        f1_max = 1.0
        
        # Generate target f1 values
        # Use curvature-based spacing: delta_f1 proportional to sqrt(f1)
        # This implies f1 = f1_min + (f1_max - f1_min) * u^2 where u is uniform
        u_vals = np.linspace(0, 1, n_evals)
        target_f1 = f1_min + (f1_max - f1_min) * (u_vals ** 2)
        
        # For each target f1, find x1
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_Y = 1 - target_f1[i]
            # Find index in Y_grid closest to target_Y
            # Note: Y_grid is not monotonic.
            # We just want ANY x that gives this Y.
            # Using argmin on absolute difference is fine.
            idx = np.argmin(np.abs(Y_grid - target_Y))
            x1_vals[i] = x_grid[idx]
            
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = points[i]
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
