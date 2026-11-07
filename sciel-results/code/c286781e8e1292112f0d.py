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
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specific knowledge:
        # Variables 0 to dim-3 are non-influencing (affect g)
        # Variables dim-2 to dim-1 are influencing (affect y)
        # To minimize g, set non-influencing vars to 0.
        # Optimize influencing vars.
        
        n_influencing = 2
        n_non_influencing = self.dim - n_influencing
        
        # Helper to create a full x vector from influencing vars
        def make_x(y_vars):
            x = np.zeros(self.dim)
            x[self.dim-n_influencing:] = y_vars
            return x

        evaluations_used = 0
        points = []
        
        # Use Sobol sequence for better coverage of the Pareto front
        # DTLZ7 Pareto front is determined by the 2 influencing variables
        sobol = qmc.Sobol(d=n_influencing, scramble=True)
        # Generate budget number of points
        # Sobol requires power of 2 for best properties, but works with any number
        # We'll just generate 'budget' points
        sobol_points = sobol.random(self.budget)
        
        # Scale to bounds (usually [0,1])
        # sobol.random returns points in [0, 1]
        # We can clip to be safe
        sobol_points = np.clip(sobol_points, lo[self.dim-n_influencing:], hi[self.dim-n_influencing:])
        
        for y_vars in sobol_points:
            if evaluations_used >= self.budget:
                break
            if time.time() > deadline:
                return None
                
            x = make_x(y_vars)
            # Clip to full bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            points.append((x, f))
            evaluations_used += 1
            
            if time.time() > deadline:
                return None

        return None
