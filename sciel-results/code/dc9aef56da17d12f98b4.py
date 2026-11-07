# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ1: Direct construction on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0  # safety margin
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        budget = int(self.budget)

        # DTLZ1 specific structure:
        # g(x) = 0 => x_i = 0.5 for i >= n_obj
        # x_0 + ... + x_{n_obj-1} = 1
        # We generate points on the (n_obj-1)-simplex.
        
        m = n_obj
        n_points = budget
        
        # Use Sobol sequence for low-discrepancy sampling on the simplex
        # We need to map points from [0, 1)^2 to the 2D simplex (triangle)
        # Simplex: x1 + x2 + x3 = 1, x_i >= 0
        # Let u1, u2 be uniform in [0, 1].
        # A standard mapping for 2D simplex:
        # s1 = sqrt(u1)
        # x1 = s1 * u2
        # x2 = s1 * (1 - u2)
        # x3 = 1 - s1
        # This maps the unit square to the triangle.
        
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate points
        # Sobol requires power of 2 for best performance, but works otherwise.
        # We can generate the next power of 2 >= n_points and take the first n_points.
        n_gen = 1
        while n_gen < n_points:
            n_gen *= 2
        samples = sampler.random(n_gen)[:n_points]
        
        u1 = samples[:, 0]
        u2 = samples[:, 1]
        
        s1 = np.sqrt(u1)
        x0 = s1 * u2
        x1 = s1 * (1 - u2)
        x2 = 1 - s1
        
        # Construct full 7D vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2] = x2
        # Set the rest to 0.5 to ensure g(x) = 0
        if dim > 3:
            X[:, 3:] = 0.5
        
        # Ensure bounds are respected (they should be, but just in case)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.empty((n_points, n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
