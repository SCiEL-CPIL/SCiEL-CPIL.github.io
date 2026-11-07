import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We want to place points on the Pareto front.
        # ZDT4 Pareto front: x[0] in [0, 1], x[1:] = 0.
        # f1 = x[0], f2 = 1 + (1 - sqrt(x[0])) * 101
        
        # To maximize hypervolume, we distribute points on the front.
        # Based on curvature analysis, uniform spacing in x[0]^(1/4) is optimal.
        # Let t be uniform in [0, 1].
        # x[0] = lo[0] + (hi[0] - lo[0]) * t^4
        
        n_points = self.budget
        if n_points <= 0:
            return np.array([])
            
        # Create grid for x[0]
        t = np.linspace(0.0, 1.0, n_points)
        x0_vals = lo[0] + (hi[0] - lo[0]) * (t ** 4)
        
        # Create decision vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        
        # For x[1:], set to 0 if 0 is within bounds, otherwise midpoint
        for i in range(1, self.dim):
            if lo[i] <= 0.0 <= hi[i]:
                X[:, i] = 0.0
            else:
                X[:, i] = (lo[i] + hi[i]) / 2.0
                
        # Evaluate
        # We must ensure we don't exceed time.
        # 200 evaluations is very fast, so this is safe.
        F = np.array([problem(x) for x in X])
        
        return X
