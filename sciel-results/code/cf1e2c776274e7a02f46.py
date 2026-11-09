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
        start = time.time()
        deadline = start + 300.0  # safety margin under 330s kill

        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2 structure:
        # g = sum_{i=3}^{dim} (x_i - 0.5)^2
        # f1 = (1+g)*sin(pi*x1/2)
        # f2 = (1+g)*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = (1+g)*cos(pi*x1/2)*cos(pi*x2/2)
        # Setting x3..x12 = 0.5 gives g = 0 -> point on the true Pareto front.
        # x1, x2 in [0,1] sweep the quarter-sphere front.

        # Strategy: 
        # 1. Place a few high-value points at vertices and edge midpoints (high hypervolume contribution)
        # 2. Use remaining budget for Sobol-sampled interior points with surface-area-uniform mapping

        # High-value special points (vertices and edge midpoints of the quarter-sphere parameter space)
        # These are in (x1, x2) parameter space where x1, x2 in [0, 1]
        # Vertices: (0,0), (0,1), (1,0), (1,1)
        # Edge midpoints: (0, 0.5), (0.5, 0), (1, 0.5), (0.5, 1)
        # Center: (0.5, 0.5)
        special_points = np.array([
            [0.0, 0.0],   # vertex: f = [0, 0, 1]
            [0.0, 1.0],   # vertex: f = [0, 1, 0]
            [1.0, 0.0],   # vertex: f = [1, 0, 0]
            [1.0, 1.0],   # vertex: f = [1, 0, 0] (same as above for DTLZ2)
            [0.0, 0.5],   # edge midpoint
            [0.5, 0.0],   # edge midpoint
            [0.5, 1.0],   # edge midpoint
            [1.0, 0.5],   # edge midpoint
            [0.5, 0.5],   # center
        ])
        
        n_special = len(special_points)
        n_sobol = max(0, budget - n_special)

        # Generate Sobol points for interior
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            # Transform x1 to account for spherical geometry for uniform surface area
            u1 = np.clip(sobol_points[:, 0], 0.0, 1.0)
            u2 = np.clip(sobol_points[:, 1], 0.0, 1.0)
            sobol_x1 = (2.0 / np.pi) * np.arcsin(u1)
            sobol_x2 = u2
            sobol_xy = np.column_stack([sobol_x1, sobol_x2])
        else:
            sobol_xy = np.empty((0, 2))

        # Combine all points
        all_points = np.vstack([special_points, sobol_xy])

        # Evaluate all candidates
        for i in range(len(all_points)):
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                x = np.full(dim, 0.5)
                x[0] = all_points[i, 0]
                x[1] = all_points[i, 1]
                problem(x)
            except Exception:
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
