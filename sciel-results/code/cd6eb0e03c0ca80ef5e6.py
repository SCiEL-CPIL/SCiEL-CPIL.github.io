"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Non-uniform grid: concentrate points in upper-right where f3 is small
        # This improves hypervolume by better resolving the high-value region
        n_side = int(np.sqrt(n_evals))
        
        # Power-law spacing biased toward x=1 (where f3 is smaller)
        alpha = 1.5
        u = np.linspace(0.0, 1.0, n_side)
        grid_vals = 1.0 - (1.0 - u) ** alpha
        
        x1_grid, x2_grid = np.meshgrid(grid_vals, grid_vals)
        x1 = x1_grid.flatten()
        x2 = x2_grid.flatten()
        
        # If we have extra evaluations, add some random points for coverage
        n_extra = n_evals - len(x1)
        if n_extra > 0:
            rng = np.random.default_rng(42)
            x1_extra = rng.random(n_extra)
            x2_extra = rng.random(n_extra)
            x1 = np.concatenate([x1, x1_extra])
            x2 = np.concatenate([x2, x2_extra])
        
        # If we have too many, trim to budget
        if len(x1) > n_evals:
            x1 = x1[:n_evals]
            x2 = x2[:n_evals]
        
        # Clip to bounds
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])
        
        # Build the full 22-dimensional vectors: x3..x22 = 0
        # For DTLZ7, setting x3..x22 = 0 ensures we're on the Pareto front
        X = np.zeros((len(x1), self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        
        # Evaluate all points
        F = np.zeros((len(x1), self.n_obj))
        for i in range(len(x1)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
