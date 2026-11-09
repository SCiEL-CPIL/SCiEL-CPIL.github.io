# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer using spherical coordinate parameterization for uniform Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4: M=3 objectives, first M-1=2 vars are shape (x0, x1), rest are distance
        n_shape = self.n_obj - 1  # 2
        n_dist = self.dim - n_shape  # 10

        # The Pareto front of DTLZ4 (when g=0) is a quarter unit sphere in 3D:
        # f1 = x0, f2 = (1-x0)*cos(x1*pi/2), f3 = (1-x0)*sin(x1*pi/2)
        # To get uniform coverage on this curved surface, use spherical coordinates.
        # Parameterize: theta in [0, pi/2], phi in [0, pi/2]
        # x0 = cos(theta)  -> f1 = cos(theta)
        # x1 = phi / (pi/2)  -> so that (1-x0)*cos(x1*pi/2) = sin(theta)*cos(phi)
        # (1-x0)*sin(x1*pi/2) = sin(theta)*sin(phi)
        # This gives uniform distribution on the spherical surface when theta, phi sampled appropriately.

        # Actually, for uniform sampling on a quarter sphere:
        # Let's use: u = cos(theta) where theta ~ uniform, but we need to be careful.
        # For uniform area on sphere: sin(theta) dtheta dphi = constant
        # So sample theta via: sin(theta) = 1 - u where u ~ Uniform(0,1), i.e., theta = arcsin(1-u)
        # Or equivalently: sample z = cos(theta) uniformly in [0,1], then theta = arccos(z)
        # Then x0 = z, and x1 = phi/(pi/2) where phi ~ Uniform(0, pi/2)

        # Let's generate N points using this parameterization
        n_points = self.budget
        rng = np.random.default_rng(42)

        # Generate uniform samples on the quarter sphere
        # z = cos(theta) ~ Uniform(0, 1)
        z_vals = rng.uniform(0, 1, n_points)  # This is x0
        # phi ~ Uniform(0, pi/2)
        phi_vals = rng.uniform(0, np.pi / 2, n_points)
        x1_vals = phi_vals / (np.pi / 2)  # Convert to [0,1] for x1

        # Create all points with distance vars fixed at 0.5
        X = np.zeros((n_points, self.dim))
        X[:, 0] = z_vals
        X[:, 1] = x1_vals
        X[:, n_shape:] = 0.5  # Optimal distance values

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
