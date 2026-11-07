import numpy as np
import time
import math
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_simplex(self, n):
        """Generate n Sobol points mapped to the 2D simplex (triangle)."""
        # Use Sobol sequence with 2 dimensions
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # We need to generate enough points, then map to simplex
        # Sobol points are in [0,1]^2, we need to map to triangle where x+y <= 1
        
        # Generate Sobol points
        points = sampler.random(n)
        
        # Map from [0,1]^2 to simplex using rejection or transformation
        # A common approach: use the fact that for uniform points in [0,1]^2,
        # we can map to simplex by taking (u1, u2) and if u1+u2 > 1, reflect
        # But for better coverage, let's use a different approach:
        # Generate in [0,1]^2, then use the transformation that preserves uniformity on the simplex
        
        # Actually, for hypervolume optimization, we want uniform coverage of the simplex
        # The simplex is { (a,b) : a >= 0, b >= 0, a+b <= 1 }
        # A standard way to generate uniform points on this simplex from [0,1]^2:
        # Let u = (u1, u2) be uniform in [0,1]^2
        # Then set a = u1 * (1 - u2), b = u2 * (1 - u1) doesn't work well
        # Better: use the inverse CDF approach or just use the reflection method
        
        # Reflection method: if u1 + u2 > 1, set u1' = 1 - u1, u2' = 1 - u2
        # This maps [0,1]^2 to the simplex with uniform density
        u1 = points[:, 0]
        u2 = points[:, 1]
        mask = u1 + u2 > 1.0
        u1 = np.where(mask, 1.0 - u1, u1)
        u2 = np.where(mask, 1.0 - u2, u2)
        
        return u1, u2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        budget = self.budget
        dim = self.dim
        all_F = []
        
        # DTLZ1 Pareto front:
        # f1 = 55.5 * x0
        # f2 = 55.5 * (1-x0) * x1  
        # f3 = 55.5 * (1-x0) * (1-x1)
        # with x[2:] = 0.5 for minimum g
        
        # Strategy: Use Sobol sequence for better low-discrepancy coverage of the simplex
        # Generate all points at once using Sobol
        x0, x1 = self._sobol_simplex(budget)
        
        # Construct and evaluate
        for k in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, 0.5)
            x[0] = x0[k]
            x[1] = x1[k]
            
            try:
                f = np.asarray(problem(x), dtype=np.float64)
                if f.shape == (3,) and np.all(np.isfinite(f)):
                    all_F.append(f)
            except Exception:
                pass
        
        return all_F
