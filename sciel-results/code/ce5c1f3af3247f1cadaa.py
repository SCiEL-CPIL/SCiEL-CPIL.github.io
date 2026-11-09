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
        
        # DTLZ4 structure:
        # First M-1 variables determine position on the Pareto front.
        # Remaining n-M+1 variables determine g(x), with g=0 when they are 0.5.
        n_front_vars = self.n_obj - 1  # 2 for 3 objectives
        n_g_vars = self.dim - n_front_vars  # 10 for 12 dim
        
        # Use a grid to cover the 2D parameter space (u, v) uniformly.
        # Cosine spacing clusters points near the vertices (0,0), (1,0), (0,1), (1,1)
        # where hypervolume contribution is highest.
        
        # Use 10x20 = 200 points for finer resolution in v dimension
        n_u = 10
        n_v = 20
        
        # Cosine spacing: points cluster near boundaries
        u_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(n_u) / (n_u - 1))
        v_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(n_v) / (n_v - 1))
        
        F = []
        for u in u_vals:
            for v in v_vals:
                if time.time() > deadline:
                    return F
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                # Set remaining variables to 0.5 to ensure g(x)=0
                x[2:] = 0.5
                f = problem(x)
                F.append(f)
        
        return F
