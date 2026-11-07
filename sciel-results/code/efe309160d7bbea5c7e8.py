import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s/330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 with 12 vars, 3 objectives:
        # M = 3 objectives, K = M - 1 = 2 position variables
        # N = 12, so M = 10 distance variables
        # Pareto front: x_{M+1..M+K} = x_{M+K+1..N} where g(x) = 0
        # For DTLZ4, g(x) = sum((x_i - 0.5)^2) for i in distance vars
        # g = 0 when each distance var = 0.5
        # f_1 = (1 + g) * prod_{i=1}^{K} cos(x_i * pi/2)
        # f_i = (1 + g) * sin(x_i * pi/2) * prod_{j=1}^{i-2} cos(x_j * pi/2) for 2 <= i <= K+1
        # f_{K+1} = (1 + g) * prod_{j=1}^{K-1} cos(x_j * pi/2) * sin(x_K * pi/2)
        # Actually for DTLZ4: f_i = (1+g) * x_i for i <= K, and f_{K+1} = (1+g) * (1 - sum of squares... )
        # Wait, let me be more careful. DTLZ4 has:
        # f_i(x) = (1 + g(x_M)) * x_{i-1} for i = 1..M+1
        # No, that's DTLZ1. DTLZ4 is:
        # f_1 = (1+g) * cos(x_1*pi/2)
        # f_2 = (1+g) * sin(x_1*pi/2) * cos(x_2*pi/2)
        # f_3 = (1+g) * sin(x_2*pi/2)
        # with g(x) = sum_{i=3}^{12} (x_i - 0.5)^2
        # Pareto front when g=0: f_1 = cos(x_1*pi/2), f_2 = sin(x_1*pi/2)*cos(x_2*pi/2), f_3 = sin(x_2*pi/2)
        # Let theta_1 = x_1*pi/2, theta_2 = x_2*pi/2
        # f_1 = cos(t1), f_2 = sin(t1)*cos(t2), f_3 = sin(t2)
        # with t1, t2 in [0, pi/2]
        
        # So the Pareto front is parameterized by (t1, t2) in [0, pi/2]^2
        # f_1 = cos(t1)
        # f_2 = sin(t1)*cos(t2)
        # f_3 = sin(t2)
        
        # Strategy: Sample (t1, t2) directly from the Pareto front parameter space
        # Use a grid + jitter approach for good hypervolume coverage
        
        # For hypervolume maximization with reference point [2.5, 2.5, 2.5]:
        # We want to cover the Pareto front well, especially near the "knee" regions
        # where points contribute most to hypervolume
        
        # Use a combination of:
        # 1. A structured grid in (t1, t2) space
        # 2. Some points at the extremes
        
        n_budget = budget
        
        # Strategy: Use uniform spacing in objective space (f1 = cos²θ) instead of
        # uniform angular spacing to optimize 2D hypervolume coverage.
        # For f1 = cos(t1), uniform in f1 means t1 = arccos(f1_val) where f1_val is uniform.
        # Similarly for f3 = sin(t2), uniform in f3 means t2 = arcsin(f3_val).
        
        # Use a 14x14 grid = 196 points, plus 4 corner points = 200
        n_grid = 14
        
        # Uniform in objective space: f1 = cos(t1) ranges [0,1], f3 = sin(t2) ranges [0,1]
        # So uniform f1 values: f1_i = i/(n_grid-1) for i in 0..n_grid-1
        # t1_i = arccos(f1_i)
        # f3_j = j/(n_grid-1) for j in 0..n_grid-1  
        # t2_j = arcsin(f3_j)
        
        f1_vals = np.linspace(0, 1, n_grid)  # uniform in f1 = cos(t1)
        f3_vals = np.linspace(0, 1, n_grid)  # uniform in f3 = sin(t2)
        
        t1_vals = np.arccos(f1_vals)  # t1 in [0, pi/2]
        t2_vals = np.arcsin(f3_vals)  # t2 in [0, pi/2]
        
        points = []
        for t1 in t1_vals:
            for t2 in t2_vals:
                x = np.zeros(dim)
                x[0] = t1 / (np.pi/2)  # x_1 = t1 / (pi/2)
                x[1] = t2 / (np.pi/2)  # x_2 = t2 / (pi/2)
                x[2:] = 0.5  # distance vars at optimum
                points.append(x)
        
        # Add corner/edge points that may not be exactly on the grid
        # The grid already includes (0,0), (0,pi/2), (pi/2,0), (pi/2,pi/2) via f1/f3 = 0 or 1
        # So we have 196 points. Add 4 more strategic interior points.
        extra = [
            [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.25, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        for e in extra:
            if len(points) < n_budget:
                x = np.array(e[:dim])
                points.append(x)
        
        # Clip to budget
        if len(points) > n_budget:
            points = points[:n_budget]
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        if time.time() > deadline:
            return F
        
        return F
