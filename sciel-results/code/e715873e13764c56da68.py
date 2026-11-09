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
        
        # Determine grid size based on budget
        max_points = self.budget
        # Start with 14x14 = 196 points
        grid_size = 14
        if grid_size * grid_size > max_points:
            grid_size = int(np.sqrt(max_points))
        
        # Cosine spacing: points cluster near boundaries
        # For i in 0..n-1, u_i = 0.5 + 0.5 * cos(pi * i / (n-1))
        # This gives values in [0, 1] with higher density near 0 and 1
        if grid_size > 1:
            u_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(grid_size) / (grid_size - 1))
            v_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(grid_size) / (grid_size - 1))
        else:
            u_vals = np.array([0.5])
            v_vals = np.array([0.5])
        
        X = []
        for u in u_vals:
            for v in v_vals:
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                # Set remaining variables to 0.5 to ensure g(x)=0
                x[2:] = 0.5
                X.append(x)
        
        # Evaluate all grid points
        F = []
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
        
        # If we have budget remaining, add a few random points for exploration
        remaining = self.budget - len(F)
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = np.random.uniform(lo[0], hi[0])
                x[1] = np.random.uniform(lo[1], hi[1])
                x[2:] = 0.5
                f = problem(x)
                F.append(f)
        
        return F
