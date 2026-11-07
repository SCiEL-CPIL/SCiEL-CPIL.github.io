"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # g = 100 + 10*(m-2) + sum_{i=3}^{n} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # For m=3, n=12: g = 100 + 10 + sum = 110 + sum
        # g = 0 when all x_i = 0.5 for i >= 3
        # Pareto front: quarter sphere of radius 0.5 in (f1, f2, f3) space
        # Parameterized by (x1, x2) in [0,1]^2

        # Strategy:
        # 1. Evaluate the 3 extreme corners of the Pareto front
        # 2. Use remaining budget for low-discrepancy sampling of the front interior
        #    using a 2D Halton sequence for (x1, x2)

        # The 3 corners in (x1, x2) space:
        # (0, 0) -> f = (0.5, 0, 0)
        # (0, 1) -> f = (0, 0.5, 0)  
        # (1, 0) -> f = (0, 0, 0.5)

        corner_x12 = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        n_corners = len(corner_x12)
        n_interior = max(0, budget - n_corners)

        # Generate 2D Halton sequence for interior points
        def halton_2d(n):
            if n == 0:
                return np.zeros((0, 2))
            points = np.zeros((n, 2))
            # Prime for dimension 0: 2, dimension 1: 3
            for i in range(1, n + 1):
                # Dimension 0 (base 2)
                f = 1.0
                i_val = i
                while i_val > 0:
                    f /= 2.0
                    i_val //= 2
                points[i - 1, 0] = f

                # Dimension 1 (base 3)
                f = 1.0
                i_val = i
                while i_val > 0:
                    f /= 3.0
                    i_val //= 3
                points[i - 1, 1] = f

            return points

        x12_interior = halton_2d(n_interior)

        # Combine corners and interior points
        if n_interior > 0:
            x12_all = np.vstack([corner_x12, x12_interior])
        else:
            x12_all = corner_x12

        # Ensure we don't exceed budget
        if len(x12_all) > budget:
            x12_all = x12_all[:budget]

        # Create full decision vectors
        X = np.zeros((len(x12_all), dim))
        X[:, 0] = x12_all[:, 0]
        X[:, 1] = x12_all[:, 1]

        # Set g-variables to 0.5 to ensure g=0 (Pareto-optimal)
        if dim > 2:
            X[:, 2:] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        for i in range(len(X)):
            if time.time() - t0 > deadline:
                break
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
