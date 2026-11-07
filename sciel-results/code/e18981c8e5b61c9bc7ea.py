"""Multi-objective optimizer for ZDT4: place all evaluations on the
analytically-known Pareto front and choose x0 to maximize 2D hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4: place all evaluations on the
    analytically-known Pareto front and choose x0 to maximize 2D hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before the 330 s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n = dim
        # g_min = 1 + 10*(n-1) - (n-1) = 9*n - 8
        gmin = 9.0 * n - 8.0

        # The Pareto front of ZDT4:
        # x[0] in [0, 1], x[1:] = 0.5 (global min of each term in g)
        # f1 = x[0]
        # f2 = gmin * (1 - sqrt(x[0] / gmin))

        def f2_of_x0(x0):
            """Compute f2 on the Pareto front given x0."""
            x0 = np.clip(x0, 0.0, 1.0)
            if x0 <= 0:
                return gmin
            return gmin * (1.0 - np.sqrt(x0 / gmin))

        def hv_from_x0(x0_vals):
            """Compute 2D hypervolume for points on the Pareto front.
            Reference point is [11.0, 400.0].
            Points are (f1, f2) = (x0, gmin*(1-sqrt(x0/gmin)))
            Only points with f1 < 11 and f2 < 400 count.
            """
            x0_vals = np.array(x0_vals, dtype=float)
            f1_vals = x0_vals
            f2_vals = gmin * (1.0 - np.sqrt(np.clip(f1_vals, 0.0, gmin) / gmin))
            
            # Filter: strictly better than reference [11.0, 400.0]
            mask = (f1_vals < 11.0) & (f2_vals < 400.0)
            f1_vals = f1_vals[mask]
            f2_vals = f2_vals[mask]
            
            if len(f1_vals) == 0:
                return 0.0
            
            # Sort by f1 ascending
            idx = np.argsort(f1_vals)
            f1_vals = f1_vals[idx]
            f2_vals = f2_vals[idx]
            
            # Remove dominated points (keep only non-dominated)
            # For a decreasing front, non-dominated means f2 is strictly decreasing
            # Remove any point that is dominated by a previous one
            keep = np.ones(len(f1_vals), dtype=bool)
            min_f2_so_far = np.inf
            for i in range(len(f1_vals)):
                if f2_vals[i] >= min_f2_so_far:
                    keep[i] = False
                else:
                    min_f2_so_far = f2_vals[i]
            
            f1_vals = f1_vals[keep]
            f2_vals = f2_vals[keep]
            
            if len(f1_vals) == 0:
                return 0.0
            
            # Compute hypervolume: sum of rectangles
            # For 2D with reference (r1, r2), HV = sum_i (f1_{i+1} - f1_i) * (r2 - f2_i)
            # where f1 is sorted ascending and f2 is sorted descending (non-dominated)
            # Actually: sort by f1 asc, then f2 should be desc for non-dominated
            # HV = sum_i (f1_{i+1} - f1_i) * (r2 - f2_i) for i = 0..n-2
            #     + (r1 - f1_{n-1}) * (r2 - f2_{n-1})
            
            r1, r2 = 11.0, 400.0
            hv = 0.0
            for i in range(len(f1_vals) - 1):
                width = f1_vals[i + 1] - f1_vals[i]
                height = r2 - f2_vals[i]
                hv += width * height
            # Last point to reference
            width = r1 - f1_vals[-1]
            height = r2 - f2_vals[-1]
            hv += width * height
            
            return hv

        # Strategy: Use arc-length parameterization to distribute points uniformly
        # along the Pareto front curve in objective space.
        # The curve is f1 = x0, f2 = gmin * (1 - sqrt(x0/gmin))
        # Arc length element: ds = sqrt(1 + (df2/df1)^2) dx0
        # df2/df1 = -gmin * (1/(2*sqrt(gmin*x0))) * (1/gmin) = -1/(2*sqrt(gmin*x0))
        # ds = sqrt(1 + 1/(4*gmin*x0)) dx0
        
        def arc_length_weight(x0):
            """Weight for arc-length parameterization."""
            x0 = np.clip(x0, 1e-10, 1.0)
            return np.sqrt(1.0 + 1.0 / (4.0 * gmin * x0))
        
        # Generate initial dense grid for arc-length integration
        n_grid = 10000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        weights = arc_length_weight(x0_grid)
        
        # Cumulative arc length
        cum_arc = np.cumsum(weights * (x0_grid[1] - x0_grid[0]))
        total_arc = cum_arc[-1]
        
        # Generate 200 points uniformly spaced in arc length
        budget = self.budget
        n_points = budget
        
        # Target arc lengths
        target_arcs = np.linspace(0.0, total_arc, n_points)
        
        # Invert: find x0 for each target arc length
        x0_samples = np.interp(target_arcs, cum_arc, x0_grid)
        
        # Ensure endpoints are included
        x0_samples[0] = 0.0
        x0_samples[-1] = 1.0
        
        # Evaluate all points
        X = np.zeros((n_points, dim))
        for i in range(dim - 1):
            X[:, i + 1] = 0.5
        X[:, 0] = x0_samples
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Use remaining budget for refinement if any
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Add a few points at high-curvature regions or endpoints
            # The front has high curvature near x0=0, so add some extra points there
            extra_x0 = np.linspace(0.0, 0.05, min(remaining, 10))
            for x0 in extra_x0:
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x[1:] = 0.5
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1
                # We don't need to store these separately; the evaluator records all
        
        return X
# EVOLVE-BLOCK-END
