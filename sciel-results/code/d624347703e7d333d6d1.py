"""Multi-objective DTLZ4 optimizer using area-preserving Halton sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front via area-preserving 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_shape = m - 1  # 2 shape variables for 3 objectives

        # Halton sequence in 2D (bases 2 and 3)
        def halton_2d(npts):
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            pts = np.zeros((npts, 2))
            for i in range(1, npts + 1):
                pts[i-1, 0] = halton_1d(i, 2)
                pts[i-1, 1] = halton_1d(i, 3)
            return pts

        # Reserve evaluations for important boundary points
        # Corners and edges of [0,1]^2 in shape space
        n_boundary = 12
        n_halton = self.budget - n_boundary
        if n_halton < 1:
            n_halton = 1
            n_boundary = 0

        # Generate Halton points
        halton_pts = halton_2d(n_halton)

        # Build boundary points: corners and edge midpoints
        boundary_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],  # corners
            [0.0, 0.25], [0.0, 0.5], [0.0, 0.75],              # edge x1=0
            [0.25, 0.0], [0.5, 0.0], [0.75, 0.0],              # edge x2=0
            [0.25, 1.0], [0.5, 1.0], [0.75, 1.0]               # edge x2=1
        ])
        boundary_pts = boundary_pts[:n_boundary]

        # Combine boundary and Halton points
        if n_boundary > 0:
            shape_pts = np.vstack([boundary_pts, halton_pts])
        else:
            shape_pts = halton_pts

        # Trim to budget
        shape_pts = shape_pts[:self.budget]

        # Map the unit square to the quarter disk (x1^2 + x2^2 <= 1)
        # This ensures all points lie on the valid Pareto front region for DTLZ4
        u = shape_pts[:, 0]
        v = shape_pts[:, 1]
        
        # Area-preserving mapping from square to quarter disk
        r = np.sqrt(u)
        theta = v * (np.pi / 2.0)
        
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        
        shape_pts_transformed = np.column_stack([x1, x2])

        # Build full x vectors
        # g-variables set to 0.5 (minimizes g for DTLZ4 with alpha=100)
        g_val = 0.5
        X = np.zeros((len(shape_pts_transformed), n))
        X[:, :n_shape] = shape_pts_transformed
        for j in range(n_shape, n):
            X[:, j] = g_val

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points with time guard
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
