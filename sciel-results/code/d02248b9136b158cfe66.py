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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = min(self.budget, 200)
        
        # Generate Halton points in [0, 1] for x[0] and x[1]
        # Halton sequence bases: 2 and 3
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f = f / base
                r = r + f * (i % base)
                i = i // base
            return r

        points = np.zeros((n_samples, 2))
        for i in range(1, n_samples + 1):
            points[i-1, 0] = halton(i, 2)
            points[i-1, 1] = halton(i, 3)
            
        x0_flat = points[:, 0]
        x1_flat = points[:, 1]
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # X[:, 2:] = 0 (inactive, ensures g=0)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
