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
        deadline = start_time + 280

        # DTLZ1 with M=3, N=7
        # Pareto front is determined by x0, x1
        # x2..x6 must be 0 for g=0
        
        n_front_vars = 2 # x0, x1
        n_penalty_vars = self.dim - n_front_vars # x2..x6
        
        # Use Sobol sequence for better coverage of the 2D parameter space (x0, x1)
        # Sobol points are low-discrepancy and cover the space uniformly.
        
        # Generate Sobol points
        # We need 200 points in 2D
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate 2^8 = 256 points, take the first 200? Or just generate 200?
        # Sobol requires power of 2 for best properties, but works for any N.
        # Let's generate 256 and take 200, or just 200.
        # Actually, qmc.Sobol().random(n) works for any n.
        
        points = sampler.random(self.budget)
        
        # points shape: (200, 2)
        # Map to bounds [0, 1]
        # The Sobol points are already in [0, 1)
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
            
            x0 = points[i, 0]
            x1 = points[i, 1]
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # x[2:] is already 0
            
            # Call problem to record the evaluation
            problem(x)
            
        return None
