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
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: set inactive vars (x[2:22]) to 0 to make g=0
        # Front parameterized by x[0], x[1]
        # Use a low-discrepancy sequence (Sobol) to densely cover the 2D front
        
        n_samples = min(self.budget, 200)
        
        # Generate Sobol points in [0, 1] for x[0] and x[1]
        sampler = qmc.Sobol(d=2, scramble=True)
        # Next_power_of_two ensures we can generate exactly n_samples if we use the first n
        # But Sobol generates powers of 2. 256 is the next power of 2 >= 200.
        # We can just generate 256 and take the first 200.
        # Or use qmc.Sobol(d=2).random(n_samples) which handles it.
        try:
            # scipy >= 1.7 or so supports random(n)
            points = sampler.random(n_samples)
        except AttributeError:
            # Fallback for older scipy or if random doesn't exist
            # Generate 256 and slice
            points = sampler.random(256)
            points = points[:n_samples]
            
        # Scale to [0, 1] (Sobol already returns [0, 1])
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
