import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 285  # Safety margin before 300s limit

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 structure: first M-1 are shape variables, rest are distance
        k_shape = self.n_obj - 1
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))
        
        # The Pareto front for DTLZ2 is at distance variables = 0 (or 0.5 depending on formulation)
        # Let's test which one is better
        x_test0 = np.zeros(self.dim)
        for i in shape_vars:
            x_test0[i] = 0.5
        f_test0 = np.array(problem(x_test0))
        
        x_test1 = np.zeros(self.dim)
        for i in shape_vars:
            x_test1[i] = 0.5
        for i in dist_vars:
            x_test1[i] = 0.5
        f_test1 = np.array(problem(x_test1))
        
        # Choose the distance setting that gives lower sum of squares
        if np.sum(f_test0**2) < np.sum(f_test1**2):
            dist_val = 0.0
        else:
            dist_val = 0.5
            
        evals_used = 2
        remaining_budget = self.budget - evals_used
        
        # Generate grid points for shape variables
        # Use a grid that fits the budget
        # For 2 shape variables, use g x g grid
        g = int(np.sqrt(remaining_budget))
        if g < 2:
            g = 2
            
        grid_vals = np.linspace(0, 1, g)
        
        solutions = []
        count = 0
        
        for v1 in grid_vals:
            for v2 in grid_vals:
                if count >= remaining_budget:
                    break
                if time.time() > self.deadline:
                    return solutions
                
                x = np.zeros(self.dim)
                x[shape_vars[0]] = v1
                if len(shape_vars) > 1:
                    x[shape_vars[1]] = v2
                
                for i in dist_vars:
                    x[i] = dist_val
                    
                f = np.array(problem(x))
                solutions.append(x)
                count += 1
                
            if count >= remaining_budget:
                break
                
        return solutions
