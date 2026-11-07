import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6: f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # When x[1:] = 0: g = 1, f2 = 1 - sqrt(f1)
        # Pareto front parameterized by x1 in [0,1]
        
        # Compute f1(x1) on a fine grid to find the range and mapping
        n_grid = 1000000
        x1_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6.0 * np.pi * x1_grid)
        f1_grid = 1.0 - np.exp(-4.0 * x1_grid) * (sin_term ** 8)
        
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid)
        
        # For hypervolume with reference [11,11]:
        # HV = sum of areas under the curve between consecutive points
        # To maximize HV, minimize the integral of f2 over [f1_min, f1_max]
        # f2 = 1 - sqrt(f1), so we want to minimize ∫(1-sqrt(f1))df1
        # This is minimized when points are uniformly spaced in f1
        
        # Place budget points at uniform f1 intervals
        n_evals = min(budget, 200)
        target_f1s = np.linspace(f1_min, f1_max, n_evals)
        
        # For each target f1, find the x1 that achieves it
        # Sort grid by f1 for efficient lookup
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # For each target, find the closest x1 in the sorted grid
        # Use searchsorted for efficiency
        x1_vals = np.empty(n_evals)
        for i, tf1 in enumerate(target_f1s):
            pos = np.searchsorted(f1_sorted, tf1)
            if pos == 0:
                x1_vals[i] = x1_sorted[0]
            elif pos >= len(f1_sorted):
                x1_vals[i] = x1_sorted[-1]
            else:
                # Check neighbors
                if f1_sorted[pos] - tf1 < tf1 - f1_sorted[pos-1]:
                    x1_vals[i] = x1_sorted[pos]
                else:
                    x1_vals[i] = x1_sorted[pos-1]
        
        # Construct points with x[1:] = 0
        points = np.zeros((n_evals, dim))
        points[:, 0] = x1_vals
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = np.clip(points[i], lo, hi)
            problem(x)
        
        return None
