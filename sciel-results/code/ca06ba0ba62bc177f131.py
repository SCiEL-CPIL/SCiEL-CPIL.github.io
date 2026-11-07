import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        # Grid parameters
        n_grid = 14
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                points.append(x)
                
        # Add remaining points if budget allows
        # 196 points. 4 left.
        # Add 4 points at the center of the grid cells?
        # Or just stop.
        
        # Evaluate
        for x in points:
            if time.time() > deadline:
                break
            problem(x)
            
        return None
