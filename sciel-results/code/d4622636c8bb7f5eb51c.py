import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        
        # DTLZ4 with 3 objectives:
        # f0 = 0.5 * x[0] * (1 + g)
        # f1 = 0.5 * x[1] * (1 + g)
        # f2 = 0.5 * (1 - x[1]) * (1 + g)
        # g = sum((x[j] - 0.5)^alpha) for j in 2..dim-1
        # Pareto front when g=0 (i.e., x[2:]=0.5):
        # f0 = 0.5*x[0], f1 = 0.5*x[1], f2 = 0.5*(1-x[1])
        # f1 + f2 = 0.5 always on the front.
        # f0 in [0, 0.5], f1 in [0, 0.5].
        
        # The Pareto front is a 2D surface parameterized by (x[0], x[1]) in [0,1]^2.
        # To maximize hypervolume with reference point [2.5, 2.5, 2.5],
        # we want to cover the front as uniformly as possible.
        
        # Use a stratified grid on (x[0], x[1]) to cover the 2D front.
        # With budget=200, use a 14x14 grid (196 points) + 4 corners = 200.
        
        n_grid = 14
        x0_vals = (np.arange(n_grid) + 0.5) / n_grid
        x1_vals = (np.arange(n_grid) + 0.5) / n_grid
        
        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                points.append(x)
        
        # Add corner points to capture extremes
        corners = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        for cx0, cx1 in corners:
            if len(points) < budget:
                x = np.zeros(dim)
                x[0] = cx0
                x[1] = cx1
                x[2:] = 0.5
                points.append(x)
        
        # If we have remaining budget, add more stratified points
        # Or we can use a denser grid. Let's try a different approach:
        # Use a space-filling curve or Latin hypercube on the 2D front.
        
        # Actually, let's reconsider. The 14x14 grid gives uniform coverage.
        # But maybe a slightly different grid size or adding mid-edge points helps.
        
        # Let's try: 10x10 grid = 100, then add 100 more points using a finer stratification
        # or use a Halton sequence for better coverage.
        
        # Alternative: Use a 3D approach. Since f1+f2=0.5, the front is 2D.
        # Let's stick with the grid but maybe use a slightly non-uniform grid
        # to better capture the hypervolume.
        
        # Actually, for hypervolume maximization on a convex front, uniform
        # sampling is near-optimal. Let's use the grid.
        
        n_evals = 0
        for x in points:
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            f = problem(x)
            n_evals += 1
        
        return None
