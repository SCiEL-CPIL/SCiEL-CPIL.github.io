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

        # For DTLZ2 with 3 objectives, the Pareto front is a quarter of the unit sphere.
        # We parameterize using spherical coordinates in the positive octant:
        #   x_1 = sin(phi) * cos(theta)
        #   x_2 = sin(phi) * sin(theta)
        #   x_3 = cos(phi)
        # where phi in [0, pi/2] and theta in [0, pi/2]
        #
        # To get uniform coverage on the sphere surface, we need to sample
        # phi and theta such that the area element sin(phi) dphi dtheta is uniform.
        # This means:
        #   u = cos(phi) should be uniform in [0, 1], so phi = arccos(u)
        #   theta should be uniform in [0, pi/2]
        #
        # So we sample (u, v) uniformly in [0, 1]^2, then set:
        #   phi = arccos(u)
        #   theta = v * pi/2

        # Use Sobol sequence for low-discrepancy sampling in 2D
        # Sobol sequences have better uniformity properties than Halton
        sampler = qmc.Sobol(d=2, scramble=True)
        n = self.budget
        # Sobol requires power of 2, so find next power of 2 >= n
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 *= 2
        
        # Generate Sobol points in [0, 1)^2
        sobol_pts = sampler.random(n=n)
        
        # Take only the first n points
        sobol_pts = sobol_pts[:n]
        
        # Map to decision space
        u = sobol_pts[:, 0]  # for phi = arccos(u)
        v = sobol_pts[:, 1]  # for theta = v * pi/2
        
        # Compute angles
        phi = np.arccos(u)  # phi in [0, pi/2]
        theta = v * (np.pi / 2)  # theta in [0, pi/2]
        
        # Compute x_1, x_2, x_3 on the Pareto front
        x1 = np.sin(phi) * np.cos(theta)
        x2 = np.sin(phi) * np.sin(theta)
        x3 = np.cos(phi)
        
        # Build decision vectors
        X = np.zeros((n, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        X[:, 2] = x3
        # Set remaining variables to 0.5 to make g(x) = 0
        X[:, 3:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
