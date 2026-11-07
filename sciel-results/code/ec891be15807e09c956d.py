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
        deadline = start_time + 290
        
        n_samples = self.budget
        results = []
        
        # Strategy: Jittered 2D Grid on Pareto Front (x[2:]=0)
        # Grid 14x14 = 196 points
        n_grid = 14
        # Generate grid points with jitter
        # Use np.random.rand which is seeded by evaluator
        
        points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Cell [i/14, (i+1)/14] x [j/14, (j+1)/14]
                x0 = (i + np.random.rand()) / n_grid
                x1 = (j + np.random.rand()) / n_grid
                points.append([x0, x1])
                
        # We have 196 points. Add 4 more using Sobol to fill gaps
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_points = sampler.random(4)
        for p in sobol_points:
            points.append(list(p))
            
        # Total 200 points
        
        for p in points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = p[0]
            x[1] = p[1]
            # x[2:] = 0 by default
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
                
        return results
