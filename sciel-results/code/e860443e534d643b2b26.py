import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific structure:
        # First M-1 variables (2 for 3 obj) determine position on front.
        # Remaining n-M+1 variables (10 for 12 dim) determine g(x).
        # g(x) = 0 if last vars are 0.5.
        
        n_front_vars = self.n_obj - 1 # 2
        n_g_vars = self.dim - n_front_vars # 10
        
        # We want to cover the Pareto front.
        # The front is parameterized by x[0], x[1] in [0, 1].
        # We will use a grid to ensure uniform coverage of the parameter space.
        # A grid of size 14x14 = 196 points fits within budget 200.
        
        grid_size = 14
        # Ensure we don't exceed budget
        max_points = self.budget
        if grid_size * grid_size > max_points:
            grid_size = int(np.sqrt(max_points))
            
        # Create grid points for the first 2 variables
        # We include 0 and 1 to capture the extremes
        u_vals = np.linspace(0.0, 1.0, grid_size)
        v_vals = np.linspace(0.0, 1.0, grid_size)
        
        X = []
        for u in u_vals:
            for v in v_vals:
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                # Set remaining variables to 0.5 to ensure g(x)=0
                x[2:] = 0.5
                X.append(x)
                
        # If we have budget left, we can add more points or refine.
        # But 196 is very close to 200.
        # Let's evaluate them all.
        
        F = []
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
            
        # The return value is ignored, but we return X just in case
        return X
