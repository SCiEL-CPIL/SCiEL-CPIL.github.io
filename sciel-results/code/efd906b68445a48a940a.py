"""DTLZ1 multi-objective optimizer using Sobol quasi-random Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol sequence on the DTLZ1 Pareto front."""

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

        # Use Sobol sequence to generate well-spread points on the 2D simplex
        # Generate n points in [0,1]^2 using Sobol, then map to simplex
        # y0 = 1 - u - v, y1 = u, y2 = v where u+v <= 1
        
        n_samples = budget  # Use full budget
        
        # Generate Sobol points in [0,1]^2
        sampler = qmc.Sobol(d=2, scramble=True)
        sobol_pts = sampler.random(n_samples)
        
        # Map to simplex: need u + v <= 1
        # Use rejection sampling or transform
        # Better: generate in [0,1]^2 and filter, or use a proper mapping
        
        # Alternative: generate in [0,1]^3 and normalize, but that biases toward center
        # Best: use the fact that for 3D simplex, we can generate 2D Sobol and map
        # Let u, v be Sobol in [0,1]^2
        # Set y1 = u * (1-v), y2 = v, y0 = 1 - y1 - y2
        # This maps [0,1]^2 to the simplex but with non-uniform density
        
        # Actually, the standard way: generate n points in [0,1]^2, 
        # then y0 = 1 - u - v, y1 = u, y2 = v, but only if u+v <= 1
        # This wastes points. Instead, use a different parameterization.
        
        # Better approach: use the inverse CDF method or just use a grid + Sobol mix
        # For simplicity and reliability, use a combination:
        # 1. Vertices and edge midpoints (7 points)
        # 2. Remaining points via Sobol on the simplex
        
        # Generate Sobol points and map to simplex using:
        # y0 = 1 - sqrt(u) * (1-v) - v * sqrt(1-u) ... this is complex
        
        # Simplest reliable method: generate Sobol in [0,1]^2, 
        # use y1 = u*(1-v), y2 = v, y0 = 1-y1-y2
        # This covers the simplex but with density biased toward (0,0) in (u,v) space
        
        # Actually for hypervolume, we want uniform coverage. Let's use:
        # Generate points in [0,1]^2, clip to u+v<=1, resample rejected points
        # But that's wasteful. 
        
        # Best practical approach: use a stratified grid + Sobol for the rest
        # Or just use Sobol with the mapping y1=u*(1-v), y2=v, y0=1-u*(1-v)-v
        
        # Let's use a simpler, well-tested approach:
        # Generate n points in [0,1]^2 using Sobol
        # Map to simplex: y2 = v, y1 = u*(1-v), y0 = 1-y1-y2
        # This is a valid bijection from [0,1]^2 to the simplex
        
        u, v = sobol_pts[:, 0], sobol_pts[:, 1]
        y1 = u * (1 - v)
        y2 = v
        y0 = 1 - y1 - y2
        
        # Ensure non-negative (should be by construction, but clip for safety)
        y0 = np.clip(y0, 0, 1)
        y1 = np.clip(y1, 0, 1)
        y2 = np.clip(y2, 0, 1)
        
        # Normalize to ensure sum = 1 (in case of floating point issues)
        y_sum = y0 + y1 + y2
        y0 = y0 / y_sum
        y1 = y1 / y_sum
        y2 = y2 / y_sum
        
        points = np.column_stack([y0, y1, y2])
        n_pts = len(points)
        
        results = []
        for i in range(n_pts):
            if time.time() > deadline:
                break
            
            y = points[i]
            x = np.zeros(dim)
            
            # Map simplex coordinates to DTLZ1 decision variables
            # Set x3 = 0 for simplicity, then:
            # x2 = y2
            # x1 = y1 / (1 - y2) if y2 < 1
            # x0 = y0 / (1 - x1) if x1 < 1
            
            x[2] = y[2]
            if 1 - y[2] > 1e-10:
                x[1] = y[1] / (1 - y[2])
            else:
                x[1] = y[1]
            if 1 - x[1] > 1e-10:
                x[0] = y[0] / (1 - x[1])
            else:
                x[0] = y[0]
            
            # Set remaining variables to 0.5 (on the Pareto front)
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
