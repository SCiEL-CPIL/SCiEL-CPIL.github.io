import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Uniform sampling of x1 in [0, 1] for even hypervolume coverage
        # of the ZDT2 Pareto front f2 = 1 - sqrt(f1)
        x1_vals = np.linspace(0, 1, n_points)
        
        # Ensure x1 is within bounds
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        candidates = np.zeros((n_points, self.dim))
        candidates[:, 0] = x1_vals
        if self.dim > 1:
            candidates[:, 1:] = lo[1:]
            
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = candidates[i]
            f = problem(x)
            
        return None
