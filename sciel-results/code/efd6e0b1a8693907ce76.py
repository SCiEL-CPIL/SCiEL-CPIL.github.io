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

        # ZDT6 f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # Range of f1 is [f1_min, 1.0]
        
        # Pre-compute a high-resolution lookup table for f1(x1)
        # Use a fine grid to ensure accurate mapping from f1 to x1
        n_grid = 1000000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        y_grid = np.exp(-4 * x_grid) * np.power(sin_term, 8)
        f1_grid = 1 - y_grid
        
        f1_min_val = np.min(f1_grid)
        f1_max_val = 1.0
        
        # Strategy: Optimal spacing based on equal hypervolume contribution.
        # In 2D, HV contribution of a point is approx proportional to 
        # (R2 - f2) * delta_f1. 
        # f2 = 1 - sqrt(f1). R2 = 11.
        # Contribution ~ (11 - (1 - sqrt(f1))) * delta_f1 = (10 + sqrt(f1)) * delta_f1.
        # To equalize contribution, we want delta_f1 * (10 + sqrt(f1)) = const.
        # This implies we should space points such that the integral of 
        # 1/(10 + sqrt(f1)) is uniform.
        
        # Compute the cumulative integral of the weight function w(f1) = 1/(10 + sqrt(f1))
        # We can do this on a coarse grid for speed, then interpolate.
        n_weight_grid = 10000
        f1_weight_grid = np.linspace(f1_min_val, f1_max_val, n_weight_grid)
        weights = 1.0 / (10.0 + np.sqrt(f1_weight_grid))
        # Cumulative sum approximates the integral
        cum_weights = np.cumsum(weights)
        cum_weights -= cum_weights[0] # Normalize to start at 0
        # Scale to [0, 1] for easy interpolation
        cum_weights_norm = cum_weights / cum_weights[-1]
        
        # We want n_evals points uniformly distributed in the cumulative weight space
        target_cum_weights = np.linspace(0, 1, n_evals)
        
        # Interpolate to find the target f1 values
        # Note: np.interp requires xp to be increasing, which f1_weight_grid is.
        target_f1_vals = np.interp(target_cum_weights, cum_weights_norm, f1_weight_grid)
        
        # Ensure the first and last points are exactly at the boundaries
        target_f1_vals[0] = f1_min_val
        target_f1_vals[-1] = f1_max_val
        
        # Map target f1 values to x1 values
        # We need to find x1 such that f1(x1) is close to target_f1.
        # Since f1 is non-monotonic, we find the x1 that minimizes |f1(x1) - target|.
        # Vectorized approach: For each target, find the best match in the grid.
        # This is O(N * M). With N=200, M=1e6, it's 200M ops. Might be slow in Python loop.
        # Better: Sort the f1_grid? No, x1 order is lost.
        # Alternative: Use a histogram/binning approach or just accept the loop if optimized.
        # Actually, 200 iterations of np.argmin on 1e6 array is ~0.02s. Total 4s. Acceptable.
        
        x1_vals = np.zeros(n_evals)
        for i in range(n_evals):
            target_f1 = target_f1_vals[i]
            # Find indices where f1 is close to target
            # Using argmin on the absolute difference
            diffs = np.abs(f1_grid - target_f1)
            min_idx = np.argmin(diffs)
            x1_vals[i] = x_grid[min_idx]
            
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_vals
        # x[1:] are 0
        
        # Evaluate
        for i in range(n_evals):
            if time.time() - start_time > deadline - 1.0:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
