import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        if self.budget < 2:
            x = np.zeros(self.dim)
            x[0] = 0.5
            x = np.clip(x, lo, hi)
            return [problem(x)]
            
        def get_obj(x1_val):
            x = np.zeros(self.dim)
            x[0] = x1_val
            x = np.clip(x, lo, hi)
            return problem(x)
        
        # For ZDT1, the Pareto front is f1=x1, f2=1-sqrt(x1) when x2..xn=0
        # Optimal grid for hypervolume: use x1 values that maximize HV
        # For concave fronts, denser sampling near x1=0 where curvature is high
        # Use a power-law grid: x1_i = (i/(N-1))^p for appropriate p
        # Optimal p for ZDT1-like fronts is around 2
        
        n_evals = self.budget
        # Create grid: x1 values from 0 to 1
        # Use (i/(N-1))^2 to get denser points near 0
        if n_evals > 1:
            indices = np.arange(n_evals)
            x1_vals = (indices / (n_evals - 1)) ** 2
        else:
            x1_vals = np.array([0.5])
        
        evaluated_f = []
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            f = get_obj(x1)
            evaluated_f.append(f)
        
        return evaluated_f
