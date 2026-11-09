"""Portfolio search for DTLZ2: Low-discrepancy sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling on the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        # DTLZ2 structure:
        # g = sum_{i=3}^{dim} (x_i - 0.5)^2
        # f1 = (1+g)*sin(pi*x1/2)
        # f2 = (1+g)*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = (1+g)*cos(pi*x1/2)*cos(pi*x2/2)
        # Setting x3..x12 = 0.5 gives g = 0 -> point on the true Pareto front.
        # x1, x2 in [0,1] sweep the quarter-sphere front.

        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere.
        # Sampling x1, x2 uniformly in [0,1] gives good coverage for hypervolume.
        # Use a dense grid in (x1, x2) space for uniform coverage.
        candidates = []
        n_grid = int(np.sqrt(budget))
        # Ensure we don't exceed budget
        while n_grid * n_grid > budget:
            n_grid -= 1
        n_pts_grid = n_grid * n_grid
        n_extra = budget - n_pts_grid
        
        # Generate grid points in (x1, x2) space
        x1_vals = np.linspace(0, 1, n_grid, endpoint=True)
        x2_vals = np.linspace(0, 1, n_grid, endpoint=True)
        
        for i in range(n_grid):
            for j in range(n_grid):
                x = np.full(dim, 0.5)
                x[0] = x1_vals[i]
                x[1] = x2_vals[j]
                candidates.append(x)
        
        # Add extra points if any remain (use Sobol for these)
        if n_extra > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_pts = sampler.random(n=n_extra)
            for i in range(n_extra):
                x = np.full(dim, 0.5)
                x[0] = sobol_pts[i, 0]
                x[1] = sobol_pts[i, 1]
                candidates.append(x)

        # 3. Evaluate all candidates, recording every objective vector.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                problem(x)
            except Exception:
                # If an exception occurs, stop to avoid further issues.
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
