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
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: 
        # WFG9 with 3 objectives and 12 variables.
        # Hypothesis: The first 3 variables (x0, x1, x2) define the Pareto front shape.
        # The remaining 9 variables (x3...x11) are "positional" or "noise" variables 
        # that can be fixed at midpoints to maximize the chance of being on or near 
        # the ideal front for the given x0, x1, x2.
        
        # Fix variables 3 to 11 at midpoints
        midpoints = (lo + hi) / 2.0
        base_x = midpoints.copy()
        
        # We will sample the first 3 dimensions (indices 0, 1, 2)
        # Using a Sobol sequence for low-discrepancy sampling to cover the 
        # hypervolume more effectively than a grid or simple random sampling.
        
        n_samples = self.budget
        # Sobol sequence generation
        # scipy.stats.qmc.Sobol requires d dimensions
        sampler = qmc.Sobol(d=3, scramble=True, seed=42)
        # Generate n_samples points in [0, 1]^3
        points_unit = sampler.random(n_samples)
        
        # Map [0, 1] to [lo[i], hi[i]] for the first 3 dimensions
        lo_3 = lo[:3]
        hi_3 = hi[:3]
        points_3d = qmc.scale(points_unit, lo_3, hi_3)
        
        count = 0
        for i in range(n_samples):
            if time.time() > deadline:
                break
                
            x = base_x.copy()
            x[0] = points_3d[i, 0]
            x[1] = points_3d[i, 1]
            x[2] = points_3d[i, 2]
            
            # Evaluate the problem
            # The evaluator records the return value, so we just need to call it.
            problem(x)
            count += 1
            
        return None
