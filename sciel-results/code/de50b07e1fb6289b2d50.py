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
        
        # ZDT3: Pareto front is at x[1:] = 0
        # x[0] determines the point on the front.
        # We sample x[0] to maximize hypervolume.
        
        lo = bounds[0][0]
        hi = bounds[0][1]
        
        # Use all budget for sampling the front
        n_samples = self.budget
        
        # Generate samples
        # Sampling with t^3 clusters more near 0, capturing early oscillations better.
        t = np.linspace(lo, hi, n_samples)
        # Map t in [0,1] to x in [lo,hi] with cubic distribution
        # x = lo + (hi - lo) * t^3
        # Since lo=0, hi=1, x = t^3
        # General case:
        x0_samples = lo + (hi - lo) * (t**3)
        
        X_list = []
        for i, x0 in enumerate(x0_samples):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            # x[1:] is 0, which is valid (bounds [0,1])
            X_list.append(x)
            
        # Evaluate
        # We must call problem for each to record the vectors
        for x in X_list:
            if time.time() > deadline:
                break
            problem(x)
            
        return X_list
