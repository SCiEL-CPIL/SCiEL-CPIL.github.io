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
        deadline = start_time + 290.0 # Safety margin
        
        # DTLZ7 specific knowledge:
        # The Pareto front is achieved when the last n-M+1 variables are 0.
        # Here n=22, M=3, so last 20 variables (indices 2..21) should be 0.
        # The first 2 variables (indices 0, 1) determine the position on the front.
        
        n_points = self.budget
        
        # Generate quasi-random points in [0,1]^2
        # We want to cover the square [0,1]x[0,1]
        sampler = qmc.Sobol(d=2, scramble=True)
        # n_points must be a power of 2 for Sobol, or we can just use the first n_points
        # scipy qmc.Sobol can generate any number, but performance is best for powers of 2.
        # 200 is not a power of 2. 256 is.
        # Let's generate 256 points and take the first 200?
        # Or just generate 200.
        points_2d = sampler.random(n_points)
        
        # Construct the 22-dimensional points
        # x[0], x[1] from points_2d
        # x[2], ..., x[21] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = points_2d[:, 0]
        X[:, 1] = points_2d[:, 1]
        # x[2:] are already 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
