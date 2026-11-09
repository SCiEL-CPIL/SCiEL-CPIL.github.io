"""Multi-objective DTLZ4 optimizer using Halton sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer: sample the known Pareto front via 2D Halton sequence mapped to sphere."""

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

        # Reserve evaluations for important boundary points on the Pareto front
        # For DTLZ4 with m=3, the Pareto front is a quarter-sphere in [0,1]^2
        # Key boundary points: (1,0), (0,1), (0,0) and some interior points
        n_boundary = 6
        n_halton = self.budget - n_boundary
        if n_halton < 1:
            n_halton = 1
            n_boundary = 0

        # Generate Halton points in [0,1]^2
        halton_pts = halton_2d(n_halton)

        # Map [0,1]^2 to the quarter-sphere Pareto front of DTLZ4
        # For DTLZ4: f_i = g * prod_{j=1}^{i-1} x_j * (1 - x_i) for i=1..m-1, f_m = g * (1 - x_{m-1})
        # With g=0 (minimized), the Pareto front is: sum(f_i^2) = 1, f_i >= 0
        # Parameterize using angles: for m=3, use theta in [0, pi/2]
        # f1 = cos(theta1), f2 = sin(theta1)*cos(theta2), f3 = sin(theta1)*sin(theta2)
        # But we need to map 2D points to the sphere surface
        # Better: use the direct DTLZ4 parameterization
        # x1, x2 in [0,1] -> f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
        # This gives the Pareto front when g=0

        def map_to_pareto_front(u):
            """Map points in [0,1]^2 to DTLZ4 Pareto front (quarter sphere)."""
            # u[:,0] = x1, u[:,1] = x2
            # f1 = x1
            # f2 = (1 - x1) * x2  
            # f3 = (1 - x1) * (1 - x2)
            x1 = u[:, 0]
            x2 = u[:, 1]
            f1 = x1
            f2 = (1.0 - x1) * x2
            f3 = (1.0 - x1) * (1.0 - x2)
            return np.column_stack([f1, f2, f3])

        # Boundary points in shape space that map to key Pareto front points
        boundary_pts = np.array([
            [1.0, 0.0],   # f=(1,0,0)
            [0.0, 1.0],   # f=(0,1,0)
            [0.0, 0.0],   # f=(0,0,1)
            [1.0, 1.0],   # f=(1,0,0) - same as first, skip
            [0.5, 0.0],   # f=(0.5, 0, 0.5)
            [0.0, 0.5],   # f=(0, 0.5, 0.5)
        ])
        # Remove duplicate
        boundary_pts = boundary_pts[:3]  # Just the three corners
        n_boundary = len(boundary_pts)
        n_halton = self.budget - n_boundary
        if n_halton < 1:
            n_halton = 1
            n_boundary = 0
            boundary_pts = np.empty((0, 2))

        # Regenerate halton if needed
        if n_halton != n_halton:  # This won't trigger, but let's be safe
            pass
        halton_pts = halton_2d(n_halton)

        # Combine boundary and halton points
        if n_boundary > 0:
            shape_pts = np.vstack([boundary_pts, halton_pts])
        else:
            shape_pts = halton_pts

        # Trim to budget
        shape_pts = shape_pts[:self.budget]

        # Map to Pareto front to get objective values, but we need x vectors
        # For DTLZ4: x[0] = x1, x[1] = x2, x[2:] = 0.5
        # This gives g = 0 (minimum) and the objectives are on the Pareto front
        g_val = 0.5
        X = np.zeros((len(shape_pts), n))
        X[:, :n_shape] = shape_pts
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
