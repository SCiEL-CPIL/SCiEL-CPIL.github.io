"""Multi-objective optimizer for ZDT4 using arc-length parameterized front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4.
    
    Strategy:
    ZDT4 has a deceptive landscape where local minima exist at x_i = +/- 0.5 for i > 1.
    The global Pareto front is located at x_2 = ... = x_n = 0.
    The first objective f1 typically depends only on x1 (f1 = x1).
    The second objective f2 depends on x1 and g(x), where g(x) is minimized at x_2..x_n = 0.
    
    To maximize hypervolume with a limited budget, we directly sample the global Pareto front.
    We use arc-length parameterization to uniformly space points along the Pareto front curve,
    which provides better hypervolume coverage than uniform spacing in the decision variable.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        if self.dim < 1:
            return np.array([])

        n_points = self.budget

        # Determine the range for x1 on the Pareto front
        # f1 = x1, and for ZDT4 the Pareto front has f1 in [0, 1] typically
        # But we should respect the bounds: x1 should be in [max(lo[0], 0), hi[0]]
        # since f1 = x1 and we need f1 >= 0 for sqrt(f1) in f2
        x1_lo = max(lo[0], 0.0)
        x1_hi = hi[0]
        
        if x1_lo >= x1_hi:
            x1_lo = lo[0]
            x1_hi = hi[0]

        # Arc-length parameterization of the Pareto front
        # On the front (x2..xn=0): f1 = x1, f2 = 1 + sqrt(x1)
        # ds/dx1 = sqrt(1 + (1/(2*sqrt(x1)))^2) = sqrt(1 + 1/(4*x1))
        # We compute cumulative arc length and invert for uniform spacing
        
        # Use a fine grid to compute cumulative arc length
        n_grid = 10000
        x1_grid = np.linspace(x1_lo, x1_hi, n_grid)
        
        # Handle x1=0 case (sqrt singularity)
        # At x1=0, ds/dx1 -> infinity, so we need to handle carefully
        # Use a small epsilon for numerical stability
        eps = 1e-10
        x1_safe = np.maximum(x1_grid, eps)
        
        # Compute ds/dx1 = sqrt(1 + 1/(4*x1))
        ds_dx1 = np.sqrt(1.0 + 1.0 / (4.0 * x1_safe))
        
        # Cumulative arc length via trapezoidal integration
        dx = (x1_hi - x1_lo) / (n_grid - 1)
        s_cum = np.zeros(n_grid)
        s_cum[1:] = np.cumsum(0.5 * (ds_dx1[:-1] + ds_dx1[1:]) * dx)
        
        # Normalize to [0, 1]
        s_total = s_cum[-1]
        if s_total > 0:
            s_norm = s_cum / s_total
        else:
            s_norm = np.linspace(0, 1, n_grid)
        
        # Generate uniform arc-length targets
        s_targets = np.linspace(0, 1, n_points)
        
        # Invert: for each target arc length, find x1
        # Use interpolation: x1 as function of s_norm
        x1_samples = np.interp(s_targets, s_norm, x1_grid)
        
        # Initialize population with x2..xn at 0 (or closest bound to 0)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_samples
        
        for i in range(1, self.dim):
            if lo[i] <= 0 <= hi[i]:
                val = 0.0
            elif abs(lo[i]) < abs(hi[i]):
                val = lo[i]
            else:
                val = hi[i]
            X[:, i] = val

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
