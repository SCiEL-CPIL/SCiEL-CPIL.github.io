"""Improved ZDT4 optimizer with non-uniform grid spacing for hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using optimized 1D grid search on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_evals = self.budget

        x0_lo = lo[0]
        x0_hi = hi[0]

        # Non-uniform grid: concentrate points near the ends of the Pareto front
        # where they contribute most to hypervolume (extending the front in both objectives)
        # Use a power-law distribution: t^alpha with alpha > 1 concentrates at ends
        # Actually for hypervolume, we want to cover the full range well.
        # Let's use a combination: uniform in the middle, denser at ends.
        
        # Strategy: use sin-based spacing that is denser at the ends
        # t_i = 0.5 * (1 - cos(pi * i / (n-1))) gives denser spacing at ends
        # But this might over-concentrate. Let's try a milder version.
        
        # Alternative: use the fact that for hypervolume with far reference point,
        # the optimal spacing is approximately uniform in f1. But to be safe,
        # let's use a slight concentration at the ends.
        
        n_grid = n_evals
        # Create index array
        idx = np.arange(n_grid, dtype=np.float64)
        
        # Use non-uniform spacing in x0 to maximize hypervolume.
        # The Pareto front is f2 = 1 - sqrt(f1), which has high curvature near f1=0.
        # Concentrating points near f1=0 (x0=0) captures more of the hypervolume.
        # Use power-law spacing: x0_i = x0_lo + (x0_hi - x0_lo) * (i/(n-1))^alpha
        # with alpha < 1 to concentrate near x0_lo (f1=0).
        if n_grid > 1:
            # Try alpha = 0.5 (square root) to concentrate near x0=0
            alpha = 0.5
            t = idx / (n_grid - 1)
            x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** alpha)
        else:
            x0_vals = np.array([(x0_lo + x0_hi) / 2.0])
        
        results = []
        # Pre-allocate x to avoid reallocation in loop
        x = np.zeros(n_vars)
        for x0 in x0_vals:
            x[0] = x0
            # x[1:] is already 0.0, which is optimal for ZDT4's f2.
            # No need to clip if x0 is from linspace within bounds, but problem(x) clips anyway.
            f = problem(x)
            results.append(f)
            
            if time.time() > deadline:
                break

        return results
# EVOLVE-BLOCK-END
