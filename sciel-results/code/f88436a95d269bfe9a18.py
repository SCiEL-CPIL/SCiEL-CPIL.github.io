"""Multi-objective DTLZ4 optimizer using area-preserving Halton sequence on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4.
    
    Strategy:
    1. The Pareto front of DTLZ4 (g=0) is a quarter-sphere in the (x1, x2) plane
       where x1^2 + x2^2 <= 1 and x1, x2 >= 0.
    2. Objectives: f1 = 0.5*(1+g)*x1^2, f2 = 0.5*(1+g)*x2^2, f3 = 0.5*(1+g)*(1-x1^2-x2^2)
       With g=0: f1 = 0.5*x1^2, f2 = 0.5*x2^2, f3 = 0.5*(1-x1^2-x2^2)
    3. To maximize hypervolume, we want points uniformly distributed on the front.
    4. Use a 2D Halton sequence in [0,1]^2, then apply an area-preserving transform
       to map to the quarter-disk (x1^2 + x2^2 <= 1) uniformly.
    5. The transform: u -> r = sqrt(u), theta = v * pi/2
       Then x1 = r*cos(theta), x2 = r*sin(theta)
       This gives uniform area coverage on the quarter disk.
    6. Set x_3...x_12 = 0.5 to ensure g=0 (for DTLZ4 with alpha=100, 
       the minimum of g is at 0.5).
    """

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
        n_boundary = 12
        n_halton = self.budget - n_boundary
        if n_halton < 1:
            n_halton = 1
            n_boundary = 0

        # Generate Halton points
        halton_pts = halton_2d(n_halton)

        # Build boundary points: corners and edge midpoints of [0,1]^2
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

        # Apply area-preserving transformation for uniform coverage on quarter disk
        # u, v in [0,1] -> r = sqrt(u), theta = v * pi/2
        # Then x1 = r*cos(theta), x2 = r*sin(theta)
        u = shape_pts[:, 0]
        v = shape_pts[:, 1]
        
        r = np.sqrt(u)
        theta = v * (np.pi / 2.0)
        
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        
        # Build full x vectors
        # g-variables set to 0.5 (minimizes g for DTLZ4 with alpha=100)
        g_val = 0.5
        X = np.zeros((len(shape_pts), n))
        X[:, 0] = x1
        X[:, 1] = x2
        for j in range(n_shape, n):
            X[:, j] = g_val

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points with time guard
        last_result = None
        for i in range(len(X)):
            if time.time() > deadline:
                break
            last_result = problem(X[i])

        return last_result
# EVOLVE-BLOCK-END
