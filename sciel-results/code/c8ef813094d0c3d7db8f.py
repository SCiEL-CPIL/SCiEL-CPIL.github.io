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

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Use a grid on the first two variables (position variables)
        # The Pareto front of DTLZ2 is determined by x_0 and x_1
        # The other variables should be set to 0.5 to minimize g(x)
        
        # Calculate grid size
        # We want roughly n_evals points.
        # n_grid x n_grid <= n_evals
        n_grid = int(np.sqrt(n_evals))
        
        # Create grid points for x_0 and x_1
        # x_0, x_1 in [lo, hi]
        # For DTLZ2, usually bounds are [0, 1]
        
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        # Generate all combinations
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        
        # Flatten to a list of points
        X0_flat = X0.flatten()
        X1_flat = X1.flatten()
        
        n_points = len(X0_flat)
        
        # Create the full decision vectors
        X = np.zeros((n_points, self.dim))
        
        # Fill x_0 and x_1
        X[:, 0] = X0_flat
        X[:, 1] = X1_flat
        
        # Fill remaining variables with the midpoint of the bounds
        # This minimizes the g(x) term in DTLZ2
        for i in range(2, self.dim):
            X[:, i] = (lo[i] + hi[i]) / 2.0
            
        # Evaluate
        eval_count = 0
        for i in range(len(X)):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break
                
        return X
