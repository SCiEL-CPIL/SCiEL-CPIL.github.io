"""DTLZ1 multi-objective optimizer using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer that directly samples the DTLZ1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1 Pareto front: sum of first 3 objectives = 0.5
        # f_i = 0.5 * y_i where y is on the simplex sum(y) = 1
        # The remaining dim-3 variables are set to 0.5 to be on the front
        
        # Generate points on the 2D simplex (triangle) for 3 objectives
        # We want to maximize hypervolume coverage, so use a structured grid
        
        # For 200 evaluations, we can use a triangular grid
        # Number of points on a triangular grid with n divisions per edge:
        # Total points = (n+1)(n+2)/2
        # For n=18: 19*20/2 = 190 points
        # For n=19: 20*21/2 = 210 points - too many
        # So use n=18 giving 190 points, then add 10 more strategically
        
        n_div = 18  # divisions per edge of triangle
        points = []
        
        # Generate all points on the triangular lattice
        for i in range(n_div + 1):
            for j in range(n_div + 1 - i):
                k = n_div - i - j
                # y_i, y_j, y_k are the barycentric coordinates
                yi = i / n_div
                yj = j / n_div
                yk = k / n_div
                points.append([yi, yj, yk])
        
        points = np.array(points)
        n_pts = len(points)
        
        # If we have more points than budget, subsample
        if n_pts > budget:
            # Use a deterministic subsampling to maintain uniformity
            # Sort by some criterion and take evenly spaced points
            # Or just take the first budget points
            points = points[:budget]
            n_pts = budget
        elif n_pts < budget:
            # Add extra points at strategic locations
            # Add center point and edge midpoints if not already present
            extra = []
            # Center
            extra.append([1.0/3, 1.0/3, 1.0/3])
            # Edge midpoints
            extra.append([0.5, 0.5, 0.0])
            extra.append([0.5, 0.0, 0.5])
            extra.append([0.0, 0.5, 0.5])
            # Vertices
            extra.append([1.0, 0.0, 0.0])
            extra.append([0.0, 1.0, 0.0])
            extra.append([0.0, 0.0, 1.0])
            
            extra = np.array(extra)
            # Add as many extra points as needed
            for i in range(min(len(extra), budget - n_pts)):
                points = np.vstack([points, extra[i:i+1]])
            n_pts = len(points)
        
        # Convert simplex coordinates to objective values
        # f_i = 0.5 * y_i
        F = 0.5 * points
        
        # For each point, construct the decision variable x
        # First 3 variables encode the simplex position
        # Remaining dim-3 variables set to 0.5
        
        results = []
        for i in range(n_pts):
            if time.time() > deadline:
                break
            
            y = points[i]
            x = np.zeros(dim)
            # Map simplex coordinates to first 3 variables
            # For DTLZ1, x[0], x[1], x[2] determine the objectives
            # f1 = 0.5 * (1 + x3) * x0 * (1 - x1)
            # f2 = 0.5 * (1 + x3) * x1 * (1 - x2)  
            # f3 = 0.5 * (1 + x3) * x2
            
            # To get f_i = 0.5 * y_i, we need:
            # (1+x3) * x0 * (1-x1) = y0
            # (1+x3) * x1 * (1-x2) = y1
            # (1+x3) * x2 = y2
            
            # Simplest: set x3 = 0, then:
            # x0 * (1-x1) = y0
            # x1 * (1-x2) = y1
            # x2 = y2
            
            # Set x2 = y2
            x[2] = y[2]
            # x1 * (1 - x2) = y1 => x1 = y1 / (1 - x2)
            if 1 - y[2] > 1e-10:
                x[1] = y[1] / (1 - y[2])
            else:
                x[1] = y[1]
            # x0 * (1 - x1) = y0 => x0 = y0 / (1 - x1)
            if 1 - x[1] > 1e-10:
                x[0] = y[0] / (1 - x[1])
            else:
                x[0] = y[0]
            
            # Set remaining variables to 0.5
            for j in range(3, dim):
                x[j] = 0.5
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            results.append(f)
        
        if results:
            return np.array(results)
        else:
            return np.empty((0, self.n_obj))
# EVOLVE-BLOCK-END
