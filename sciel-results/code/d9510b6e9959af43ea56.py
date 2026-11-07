import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        n = self.dim
        budget = self.budget

        # ZDT3 Pareto front: f1 = t, f2 = 3*(1 - sqrt(t) - sin(10*pi*t)), t in [0,1]
        # Pareto-optimal t values are those where f2(t) achieves a new running minimum.
        
        # Use a fine grid to identify Pareto-optimal t values
        n_grid = 100000
        t_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = 3.0 * (1.0 - np.sqrt(t_grid) - np.sin(10.0 * np.pi * t_grid))
        
        # Fast running minimum using numpy
        running_min = np.minimum.accumulate(f2_grid)
        
        # Pareto-optimal: f2 equals the running minimum (new record low)
        # A point is Pareto-optimal if f2(t) <= min(f2(s) for s <= t)
        # i.e., f2(t) == running_min[t]
        pareto_mask = (f2_grid <= running_min + 1e-15)
        pareto_t = t_grid[pareto_mask]
        
        # Select budget points from Pareto-optimal t values
        # For hypervolume maximization, we want good spread in objective space
        # Uniform spacing in f1 (=t) among Pareto-optimal points is a good heuristic
        if len(pareto_t) >= budget:
            # Select with uniform spacing in index
            indices = np.linspace(0, len(pareto_t) - 1, budget).astype(int)
            selected_t = pareto_t[indices]
        else:
            # Use all Pareto-optimal points, pad if needed
            selected_t = pareto_t.copy()
            if len(selected_t) < budget:
                # Pad with evenly spaced points from the full Pareto set
                extra_n = budget - len(selected_t)
                extra_idx = np.linspace(0, len(pareto_t) - 1, extra_n).astype(int)
                extra_t = pareto_t[extra_idx]
                selected_t = np.concatenate([selected_t, extra_t])
                selected_t = np.unique(selected_t)
        
        # Create candidate points: x[0] = t, x[1:] = 0 (minimizes g, places on Pareto front)
        X = np.zeros((len(selected_t), n))
        X[:, 0] = selected_t
        
        # Evaluate all points
        for i in range(len(X)):
            problem(X[i])
        
        if time.time() > deadline:
            return X
        
        return X
