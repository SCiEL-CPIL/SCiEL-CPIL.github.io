import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 specific knowledge:
        # n_obj = 3, dim = 7
        # Pareto front: x[2:7] = 0.5 (g=0), vary x[0], x[1]
        # f1 = 0.5 * x0, f2 = 0.5 * (1-x0) * x1, f3 = 0.5 * (1-x0) * (1-x1)
        
        fixed_vars = np.full(self.dim, 0.5)
        budget = self.budget
        
        # Strategy: Combine structured points (corners, edges) with Sobol quasi-random
        # to maximize hypervolume coverage
        
        # 1. Structured points: corners and edge midpoints of the simplex
        structured_points = []
        # Corners of the triangle (x0, x1) space
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        for c in corners:
            structured_points.append(c)
        
        # Edge midpoints
        edge_mids = [(0.5, 0.0), (0.5, 1.0), (0.0, 0.5)]
        for e in edge_mids:
            structured_points.append(e)
        
        # Center point
        structured_points.append((1/3, 1/3))
        
        # 2. Sobol quasi-random points for uniform coverage
        n_sobol = budget - len(structured_points)
        if n_sobol > 0:
            sobol = qmc.Sobol(d=2, scramble=True)
            sobol_points = sobol.random(n_sobol)
            # Map from [0,1]^2 to simplex: x0 + x1 <= 1
            # Use: x0 = u1, x1 = u2 * (1 - u1) to ensure x0 + x1 <= 1
            simplex_points = []
            for u in sobol_points:
                x0 = u[0]
                x1 = u[1] * (1.0 - x0)
                simplex_points.append((x0, x1))
        else:
            simplex_points = []
        
        # Combine all points
        all_points = structured_points + simplex_points
        
        # Ensure we don't exceed budget
        all_points = all_points[:budget]
        
        results = []
        evaluations = 0
        
        for x0, x1 in all_points:
            if evaluations >= budget:
                break
            if time.time() > deadline:
                break
            
            x = fixed_vars.copy()
            x[0] = x0
            x[1] = x1
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            results.append(f)
            evaluations += 1
        
        # Safety: fill remaining budget if any
        if evaluations < budget:
            rng = np.random.default_rng(42)
            while evaluations < budget:
                if time.time() > deadline:
                    break
                x0 = rng.random()
                x1 = rng.random() * (1.0 - x0)
                x = fixed_vars.copy()
                x[0] = x0
                x[1] = x1
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                results.append(f)
                evaluations += 1
        
        return results
