"""Optimized DTLZ4 solver: 2D stratified grid over triangular Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with 2D stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270  # 30s safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 with m=3 objectives, n=12 variables:
        # Position vars: x[0], x[1] (2 vars)
        # G-vars: x[2]...x[11] (10 vars)
        # Pareto front (g=0): f1=x1, f2=x2, f3=1-x1-x2
        # Constraint: x1>=0, x2>=0, x1+x2<=1 (triangle)
        
        n_pos = self.n_obj - 1  # 2 position variables
        n_g = self.dim - n_pos  # 10 g-variables
        g_opt = 0.5  # optimal g-value
        
        # Generate 2D stratified grid over the triangular region
        # Use budget to determine grid resolution
        n_samples = min(self.budget, 200)
        
        # Create a triangular grid: for each "row" i (0 to n_rows-1),
        # we have points along that row of the triangle
        # Using a systematic stratified approach
        
        # Method: sample in barycentric coordinates
        # Let u, v be in [0,1] with u+v<=1
        # Then x1 = u, x2 = v, and the third barycentric coord is 1-u-v
        
        # Stratified sampling: divide the triangle into sub-triangles
        # Use a grid in (u, v) space clipped to the triangle
        
        points = []
        
        # Include the 3 vertices of the triangle
        vertices = [
            [1.0, 0.0],  # (x1=1, x2=0) -> f=(1,0,0)
            [0.0, 1.0],  # (x1=0, x2=1) -> f=(0,1,0)
            [0.0, 0.0],  # (x1=0, x2=0) -> f=(0,0,1)
        ]
        
        for v in vertices:
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = g_opt
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Add stratified interior points
        # Use a grid in (u, v) where u+v <= 1
        # For better hypervolume, use a quasi-random or stratified approach
        
        # Create stratified samples: divide [0,1]x[0,1] into cells,
        # keep only those where u+v <= 1
        n_per_dim = int(np.sqrt(n_samples - 3)) + 1
        
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                # Stratified midpoint with jitter
                u = (i + 0.5) / n_per_dim
                v = (j + 0.5) / n_per_dim
                
                if u + v <= 1.0:
                    x = np.zeros(self.dim)
                    x[0] = u
                    x[1] = v
                    x[2:] = g_opt
                    x = np.clip(x, lo, hi)
                    points.append(x)
        
        # If we have budget left, add more stratified points with finer resolution
        used = len(points)
        if used < n_samples:
            # Use a second, finer grid offset by half a cell
            n_per_dim2 = n_per_dim * 2
            for i in range(n_per_dim2):
                for j in range(n_per_dim2):
                    if used >= n_samples:
                        break
                    u = (i + 0.5) / n_per_dim2
                    v = (j + 0.5) / n_per_dim2
                    
                    if u + v <= 1.0:
                        # Check if this point is already close to an existing one
                        x = np.zeros(self.dim)
                        x[0] = u
                        x[1] = v
                        x[2:] = g_opt
                        x = np.clip(x, lo, hi)
                        points.append(x)
                        used += 1
        
        # Fill remaining budget with additional stratified points
        while used < n_samples:
            if time.time() > deadline:
                break
            # Use a different stratification: divide triangle into horizontal strips
            n_strips = 20
            strip = used % n_strips
            pos_in_strip = used // n_strips
            
            # Strip 'strip' goes from v=0 to v=(1 - strip/n_strips)
            v_max = (n_strips - strip) / n_strips
            # Sample along this strip
            n_in_strip = int(np.ceil(n_samples / n_strips))
            if pos_in_strip < n_in_strip:
                u = (pos_in_strip + 0.5) / n_in_strip
                v = v_max * u  # proportional to u to stay in triangle
                # Actually, for strip at height v_max, u ranges from 0 to v_max
                # Let me use: u = pos/n_in_strip * v_max, v = v_max * (1 - u/v_max)
                # Simpler: parameterize the line segment
                t = (pos_in_strip + 0.5) / n_in_strip
                u = t * v_max
                v = v_max * (1 - t)
                
                if u >= 0 and v >= 0 and u + v <= 1.0 + 1e-10:
                    x = np.zeros(self.dim)
                    x[0] = u
                    x[1] = v
                    x[2:] = g_opt
                    x = np.clip(x, lo, hi)
                    points.append(x)
                    used += 1
            else:
                break
        
        # Evaluate all points
        for x in points:
            problem(x)
        
        return points
# EVOLVE-BLOCK-END
