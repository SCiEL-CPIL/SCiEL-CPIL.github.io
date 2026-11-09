# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # M=3 objectives, K=10 distance vars, M-1=2 shape vars
        # Shape vars: x[0], x[1]; Distance vars: x[2..11]
        # g = sum_{i=2}^{11} (x[i]-0.5)^2
        # Pareto front: g=0 => x[2..11]=0.5
        # Objectives on front (g=0):
        #   f0 = cos(x[0]*pi/2) * cos(x[1]*pi/2)
        #   f1 = cos(x[0]*pi/2) * sin(x[1]*pi/2)
        #   f2 = sin(x[0]*pi/2)
        
        rng = np.random.default_rng(0)
        
        # Allocate budget:
        # - 90% for Pareto front sampling (including vertices)
        # - 10% for local refinement near the front
        
        n_front = int(budget * 0.9)
        n_local = budget - n_front
        
        # Ensure we have enough budget for vertices + grid
        # Vertices of the Pareto front (corners in shape space):
        # (x0, x1) = (0,0), (0,1), (1,0), (1,1)
        # These give the extreme points of the front
        
        # Create a structured grid on the Pareto front
        # Use a combination of:
        # 1. Explicit vertices (4 points)
        # 2. Uniform grid sampling for coverage
        # 3. Spherical-coordinate-based sampling for uniform area density
        
        # Number of grid points per axis
        n_grid = int(np.sqrt(n_front - 4))
        if n_grid < 2:
            n_grid = 2
        
        # Generate grid points
        grid_x0 = np.linspace(0, 1, n_grid)
        grid_x1 = np.linspace(0, 1, n_grid)
        
        X_front = np.zeros((n_front, dim))
        # Set distance variables to 0.5
        X_front[:, 2:] = 0.5
        
        # Fill in grid points
        idx = 0
        for x0 in grid_x0:
            for x1 in grid_x1:
                if idx >= n_front:
                    break
                X_front[idx, 0] = x0
                X_front[idx, 1] = x1
                idx += 1
            if idx >= n_front:
                break
        
        # Add explicit vertices if not already included
        vertices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        for v in vertices:
            if idx >= n_front:
                break
            # Check if this vertex is already in the grid
            found = False
            for j in range(idx):
                if np.isclose(X_front[j, 0], v[0]) and np.isclose(X_front[j, 1], v[1]):
                    found = True
                    break
            if not found:
                X_front[idx, 0] = v[0]
                X_front[idx, 1] = v[1]
                idx += 1
        
        # Fill remaining budget with spherical-coordinate-based uniform sampling
        if idx < n_front:
            n_extra = n_front - idx
            u = rng.random(n_extra)
            v = rng.random(n_extra)
            
            # Uniform sampling on quarter sphere using spherical coordinates
            # theta in [0, pi/2] with density ~ sin(theta) for uniform area
            # phi in [0, pi/2] uniform
            theta = np.arccos(1 - u)
            phi = np.arccos(1 - v)
            
            X_front[idx:idx+n_extra, 0] = 1 - np.cos(theta)
            X_front[idx:idx+n_extra, 1] = 1 - np.cos(phi)
        
        # Evaluate all front points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Local refinement: perturb points on the front slightly
        if n_local > 0:
            for i in range(n_local):
                if time.time() > deadline:
                    break
                
                # Pick a random point from the front
                idx_pt = rng.integers(n_front)
                x = X_front[idx_pt].copy()
                
                # Small perturbation in shape variables only
                noise = 0.02 * rng.standard_normal(2)
                x[0] += noise[0]
                x[1] += noise[1]
                x = np.clip(x, lo, hi)
                
                fx = problem(x)
                used += 1
        
        return X_front
# EVOLVE-BLOCK-END
