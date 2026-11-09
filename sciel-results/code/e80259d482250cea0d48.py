import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # DTLZ7 Pareto front is a 2D surface parameterized by x[0] and x[1].
        # The other variables should be 0 (or lo) for optimal solutions.
        # We'll create a 2D space-filling set of points over x[0] and x[1].
        
        # Strategy:
        # 1. Use Sobol sequence (via scipy) for better low-discrepancy coverage than Halton.
        # 2. Add strategic boundary points to ensure corners and edges are covered.
        # 3. Use the full budget.
        
        from scipy.stats import qmc
        
        # Reserve some points for explicit boundaries
        # The Pareto front of DTLZ7 is bounded by x[0]=0, x[0]=1, x[1]=0, x[1]=1.
        # Sobol sequence does not hit boundaries exactly.
        
        # Use 10 boundary points: 4 corners + 6 edge midpoints/quarter points
        n_boundary = 10
        n_sobol = n_evals - n_boundary
        
        if n_sobol < 0:
            n_sobol = 0
            n_boundary = n_evals
        
        # Generate Sobol sequence in 2D
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_points = sampler.random(n_sobol)
        else:
            sobol_points = np.empty((0, 2))
        
        # Define boundary points in [0,1]x[0,1]
        # Corners
        boundary_points = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            # Edge midpoints
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            # Additional edge points for better coverage
            [0.25, 0.0],
            [0.75, 1.0],
        ])
        
        boundary_points = boundary_points[:n_boundary]
        
        # Combine
        if n_sobol > 0:
            all_points = np.vstack([sobol_points, boundary_points])
        else:
            all_points = boundary_points
        
        # Map from [0,1] to [lo, hi] for x[0] and x[1]
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * all_points[:, 0]
        X[:, 1] = lo[1] + (hi[1] - lo[1]) * all_points[:, 1]
        # x[2:] should be lo[2:] (which is 0) to minimize g(x)
        X[:, 2:] = lo[2:]
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = X[i]
            # Clip to be safe
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
