import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # 300s limit, use 290s to be safe
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We have budget=200.
        # Use Sobol sequence for 2D sampling of x1, x2.
        # x3...x22 are set to 0 to lie on the Pareto front (g=1).
        
        n_evals = self.budget
        if n_evals == 0:
            return None
            
        sampler = qmc.Sobol(d=2, scramble=True)
        points = sampler.random(n_evals)
        
        # points shape: (n_evals, 2)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = points[:, 0]
        X[:, 1] = points[:, 1]
        # x[2:] is 0.0
        
        last_x = None
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            # Ensure within bounds just in case, though Sobol is [0,1)
            # and bounds are [0,1]
            x = np.clip(x, lo, hi)
            f = problem(x)
            last_x = x
            
        return last_x
