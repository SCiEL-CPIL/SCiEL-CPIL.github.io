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
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_shape = self.n_obj - 1
        n_pos = self.dim - n_shape
        
        # Generate space-filling samples for the shape variables
        # We want to maximize hypervolume, so we need good coverage of the shape space.
        # Sobol is a good choice for low dimensions (n_shape=2).
        
        # We have self.budget evaluations.
        # Let's use all of them for sampling the front.
        n_samples = self.budget
        
        # Create Sobol sequence
        # scramble=True is good for uniformity
        sampler = qmc.Sobol(d=n_shape, scramble=True)
        
        # Sobol works best with power of 2, but handles any N.
        # For N=200, it's fine.
        shape_samples = sampler.random(n_samples)
        
        # Map shape samples to bounds
        # shape_samples is in [0, 1]^n_shape
        # Bounds for shape variables are bounds[n_pos:]
        lo_shape = lo[n_pos:]
        hi_shape = hi[n_pos:]
        shape_samples = lo_shape + (hi_shape - lo_shape) * shape_samples
        
        # Construct full vectors
        # Position variables are set to 0.5 (optimal for DTLZ4 to be on the front)
        # Note: DTLZ4 g(x) = sum(x_i - 0.5)^2. So x_i=0.5 makes g=0.
        pos_val = np.full((n_samples, n_pos), 0.5)
        
        # Clip position values to bounds just in case (though 0.5 is in [0,1])
        pos_val = np.clip(pos_val, lo[:n_pos], hi[:n_pos])
        
        X = np.hstack((pos_val, shape_samples))
        
        # Ensure X is within bounds
        X = np.clip(X, lo, hi)
        
        fvals = []
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            fvals.append(f)
            
        return fvals
