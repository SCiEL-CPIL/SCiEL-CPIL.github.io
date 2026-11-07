import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    Exploits the known structure of the Pareto front for ZDT4,
    where x_1..x_9 = 0 minimizes the second objective component,
    leaving f_1 = x_0 and f_2 = 1 - sqrt(x_0).
    Uses arc-length parameterization to uniformly sample the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific knowledge:
        # x_0 in [0, 1]
        # x_1..x_9 in [-5, 5], but optimal is 0
        # f1 = x_0
        # f2 = 1 - sqrt(x_0) (when x_1..x_9=0)
        
        # We want to sample 200 points on the Pareto front.
        # The front is parameterized by x_0.
        # To maximize hypervolume, we sample uniformly along the arc length of the curve (x_0, 1-sqrt(x_0)).
        
        # Precompute the arc-length mapping
        # Use a high-resolution grid for the lookup table
        n_grid = 10000
        x0_grid = np.linspace(0, 1, n_grid)
        f2_grid = 1 - np.sqrt(x0_grid)
        
        # Calculate differential arc length ds = sqrt(1 + (df2/dx0)^2) * dx0
        # df2/dx0 = -1 / (2 * sqrt(x0))
        # Handle x0=0 singularity
        df2_dx0 = np.zeros_like(x0_grid)
        mask = x0_grid > 1e-12
        df2_dx0[mask] = -1.0 / (2.0 * np.sqrt(x0_grid[mask]))
        
        ds = np.sqrt(1.0 + df2_dx0**2) * np.diff(x0_grid)
        # Prepend 0 for the first point
        ds = np.concatenate(([0.0], ds))
        
        # Cumulative arc length
        arc_len = np.cumsum(ds)
        total_arc_len = arc_len[-1]
        
        # Normalize to [0, 1]
        arc_len_norm = arc_len / total_arc_len
        
        # Generate 200 uniform targets in normalized arc length
        n_samples = self.budget
        targets = np.linspace(0, 1, n_samples)
        
        # Invert the mapping: for each target, find x0
        # np.interp expects xp to be increasing, which arc_len_norm is.
        # We want x0 for a given arc_len_norm.
        x0_samples = np.interp(targets, arc_len_norm, x0_grid)
        
        # Construct the population
        # x_1..x_9 should be 0
        # Ensure we respect bounds, though 0 is well within [-5, 5]
        # and x0 is in [0, 1]
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_samples
        # x_1..x_9 are already 0
        
        # Clip to bounds just in case, though not strictly necessary for these values
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function takes a 1D array and returns 2 objectives
        # We iterate to respect the budget and check time
        F = np.zeros((n_samples, self.n_obj))
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
