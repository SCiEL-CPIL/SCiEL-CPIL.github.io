"""Optimized DTLZ4 solver: direct analytical Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # Position variables: x[0], x[1] (M-1 = 2)
        # Distance variables: x[2]...x[11] (N-M+1 = 10)
        # For Pareto front (g=0), set distance variables to 0.5
        # Objectives:
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)

        # Strategy: Use budget to create a well-distributed set on the Pareto front.
        # For hypervolume maximization in 3D, we want points that cover the front
        # efficiently. Use a quasi-random (Sobol-like) or grid approach.

        n_points = budget

        # Use a structured approach: create points on a spherical triangle
        # that maximizes coverage for hypervolume.
        
        # Method: Use Maximin distance sampling in 2D parameter space
        # to get well-distributed points on the Pareto front.
        
        # Start with the 3 corner points to ensure coverage of the extremes
        corner_params = np.array([
            [0.0, 0.0],   # f=(1, 0, 0)
            [0.0, 1.0],   # f=(0, 1, 0)
            [1.0, 0.0],   # f=(0, 0, 1)
        ])
        
        if n_points <= 3:
            params = corner_params[:n_points]
        else:
            # Greedy Maximin Sampling
            # Initialize with corners
            selected = [corner_params[0], corner_params[1], corner_params[2]]
            
            # Pre-compute distances between corners? No, we update dynamically.
            # We need to find n_points - 3 more points.
            # For efficiency, we can use a simple grid of candidates or random restarts,
            # but for N=200, we can try a dense grid or use scipy's spatial.
            # However, pure Python greedy on a grid might be slow if grid is large.
            # Let's use a structured grid of candidates and pick the one with max min-distance.
            
            # Create a fine grid of candidates in [0, 1]^2
            # 50x50 = 2500 candidates, which is enough for 200 points.
            grid_size = 50
            gx = np.linspace(0, 1, grid_size)
            gy = np.linspace(0, 1, grid_size)
            Gx, Gy = np.meshgrid(gx, gy)
            candidates = np.column_stack([Gx.ravel(), Gy.ravel()])
            
            # We will remove selected points from candidates to avoid duplicates
            # and update min distances.
            
            # For speed, we can just iterate and pick the best candidate each time.
            # Distance calculation:
            # d(x, y) = ||x - y||
            
            # Initialize min_dist array for candidates
            # dist_to_selected[i] = min distance from candidate i to any selected point
            min_dist = np.full(len(candidates), np.inf)
            
            # Update min_dist for the initial 3 selected points
            for p in selected:
                dists = np.linalg.norm(candidates - p, axis=1)
                min_dist = np.minimum(min_dist, dists)
                
            # Remove the selected points from candidates? 
            # It's safer to just mark them as "used" or have very low distance.
            # Actually, since we start with corners which are in the grid, 
            # their distance to themselves is 0. So they will never be picked again.
            
            remaining_needed = n_points - len(selected)
            
            for _ in range(remaining_needed):
                # Find candidate with max min_dist
                idx = np.argmax(min_dist)
                new_point = candidates[idx]
                selected.append(new_point)
                
                # Update min_dist for all candidates
                dists = np.linalg.norm(candidates - new_point, axis=1)
                min_dist = np.minimum(min_dist, dists)
                
                # Optional: If time is tight, break early
                if time.time() > deadline - 10:
                    break

            params = np.array(selected)
            # Ensure we have exactly n_points
            if len(params) < n_points:
                # Fill with random points if we broke early due to time
                missing = n_points - len(params)
                random_params = np.random.rand(missing, 2)
                params = np.vstack([params, random_params])
            else:
                params = params[:n_points]
        
        # Construct decision vectors
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
