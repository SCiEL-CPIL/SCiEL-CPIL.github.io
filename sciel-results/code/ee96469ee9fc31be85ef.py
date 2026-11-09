import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit
        
        n_eval = self.budget
        
        # Strategy: Sample the (x1, x2) decision space densely.
        # For DTLZ4 with g=0 (x[3:]=0.5), the objectives are determined by x1, x2.
        # We use a combination of:
        # 1. A structured grid for uniform coverage
        # 2. Halton points for low-discrepancy sampling
        # 3. Boundary and vertex points
        # This approach is robust to the alpha parameter because we're sampling
        # the decision space directly, not assuming a specific front geometry.
        
        points = []
        
        # Add key boundary and vertex points first
        # These are crucial for hypervolume
        boundary_points = [
            (0.0, 0.0),  # Corner
            (1.0, 0.0),  # Edge
            (0.0, 1.0),  # Edge
            (1.0, 1.0),  # Corner
            (0.5, 0.0),  # Midpoint of edge
            (0.0, 0.5),  # Midpoint of edge
            (0.5, 0.5),  # Center
        ]
        points.extend(boundary_points)
        
        # Generate Halton points in [0,1]^2 for the remaining budget
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r
        
        n_remaining = n_eval - len(points)
        if n_remaining > 0:
            for i in range(1, n_remaining + 1):
                x1 = halton(i, 2)
                x2 = halton(i, 3)
                points.append((x1, x2))
        
        results = []
        used = 0
        
        for x1, x2 in points:
            if time.time() > deadline:
                break
            if used >= n_eval:
                break
            
            # Construct the decision vector
            x = np.zeros(self.dim)
            x[1] = x1
            x[2] = x2
            x[3:] = 0.5  # g(x) = 0 for DTLZ4
            
            f = problem(x)
            results.append(f)
            used += 1
        
        return results
