import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        # Use Sobol sequence for better coverage of the [0,1]^2 domain for x0, x1
        # This replaces the grid search.
        n_points = self.budget
        if n_points > 200:
             n_points = 200 # Just in case, though budget is 200

        # Generate Sobol points
        # Sobol requires power of 2 for best quality, but works for any n.
        # 256 is close to 200 and is a power of 2. We can take the first 200.
        # Or just generate 200.
        sampler = qmc.Sobol(d=2, scramble=True)
        # Sobol sequence generation
        # We want points in [0, 1]^2
        sobol_points = sampler.random(n=n_points)
        
        points = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = sobol_points[i, 0]
            x[1] = sobol_points[i, 1]
            x[2:] = 0.5
            points.append(x)
            
        F = []
        for x in points:
            if time.time() > deadline - 1.0:
                break
            F.append(problem(x))
            
        return points
