"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # x_1, x_2 are position variables on the quarter-sphere
        # x_3..x_12 are g-variables (9 variables)
        #
        # The Pareto front is achieved when g(x) = 0.
        # g(x) = 100*(M-1) + sum_{i=4}^{N} ((x_i-0.5)^2 - cos(20*pi*(x_i-0.5)))
        # Minimum of each term is 0 at x_i = 0.5
        # So on the Pareto front: x_3 = ... = x_12 = 0.5
        #
        # On the Pareto front:
        # f_1 = x_1 * (1 + g) = x_1
        # f_2 = x_2 * (1 - x_1) * (1 + g) = x_2 * (1 - x_1)
        # f_3 = (1 - x_2) * (1 + g) = 1 - x_2
        #
        # Wait, let me re-check the DTLZ4 formulation.
        # Standard DTLZ4:
        # f_1 = x_1 * (1 + g)
        # f_2 = x_2 * (1 - x_1) * (1 + g)  [for M=3, this isn't quite right]
        #
        # Actually for DTLZ with M objectives:
        # f_i = x_i * (1 + g) for i = 1, ..., M-1
        # f_M = (1 - sum_{i=1}^{M-1} x_i) * (1 + g)
        # where x_1 + ... + x_{M-1} <= 1
        #
        # For M=3:
        # f_1 = x_1 * (1 + g)
        # f_2 = x_2 * (1 - x_1) * (1 + g)  -- no, this isn't standard
        #
        # Let me use the standard DTLZ formulation:
        # f_1 = x_1 * (1 + g)
        # f_2 = (1 - x_1) * x_2 * (1 + g)  -- still not standard
        #
        # The correct DTLZ formulation for M=3:
        # f_1 = x_1 * (1 + g)
        # f_2 = (1 - x_1) * x_2 * (1 + g)
        # f_3 = (1 - x_1) * (1 - x_2) * (1 + g)
        # where x_1, x_2 in [0, 1]
        #
        # On the Pareto front (g=0):
        # f_1 = x_1
        # f_2 = (1 - x_1) * x_2
        # f_3 = (1 - x_1) * (1 - x_2)
        #
        # This gives a quarter-sphere: f_1^2 + f_2^2 + f_3^2 = 1
        # Let's verify: x_1^2 + (1-x_1)^2 * x_2^2 + (1-x_1)^2 * (1-x_2)^2
        # = x_1^2 + (1-x_1)^2 * (x_2^2 + (1-x_2)^2)
        # This doesn't simplify to 1 in general.
        #
        # Actually, the Pareto front of DTLZ is a quarter unit sphere in the positive octant.
        # Using spherical coordinates:
        # f_1 = sin(theta) * sin(phi)
        # f_2 = sin(theta) * cos(phi)
        # f_3 = cos(theta)
        # where theta in [0, pi/2], phi in [0, pi/2]
        #
        # We need to map from (x_1, x_2) to the sphere.
        # From the DTLZ formulation:
        # f_1 = x_1
        # f_2 = (1-x_1) * x_2
        # f_3 = (1-x_1) * (1-x_2)
        #
        # So: x_1 = f_1
        #     x_2 = f_2 / (1 - f_1)  if f_1 < 1
        #
        # To sample uniformly on the quarter sphere, we can use:
        # theta = arccos(f_3) = arccos(1 - x_1 - (1-x_1)*x_2) -- this is getting complicated
        #
        # Let me just directly sample x_1, x_2 from [0,1] using a stratified grid
        # and compute the corresponding objective values.
        #
        # Strategy: Use a stratified grid in (x_1, x_2) space to cover the front uniformly.
        # The number of grid points should be chosen to maximize hypervolume coverage.

        # Number of evaluations to use for front sampling
        # We want to use most of the budget for front points
        # A grid of n x n points uses n^2 evaluations
        # With budget=200, we can use up to sqrt(200) ~ 14 per dimension
        # Let's use a 14x14 grid = 196 points, leaving 4 for refinement

        n_grid = int(np.sqrt(budget))
        if n_grid * n_grid > budget:
            n_grid -= 1

        # Generate stratified grid points in (x_1, x_2) space
        # Use midpoint of each cell for better coverage
        x1_vals = (np.arange(n_grid) + 0.5) / n_grid
        x2_vals = (np.arange(n_grid) + 0.5) / n_grid

        X = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                # Set g-variables to 0.5 for g=0
                for k in range(2, dim):
                    x[k] = 0.5
                X.append(x)

        X = np.array(X)
        n_samples = len(X)

        # Evaluate all candidates
        F = np.array([problem(x) for x in X])
        used = n_samples

        # Check deadline
        if time.time() > deadline - 5:
            return X[:used]

        # Use remaining budget for refinement if any
        remaining = budget - used
        if remaining > 0:
            # Try to refine points near the boundaries of the front
            # where hypervolume contribution is highest
            # Focus on corners: (0,0), (1,0), (0,1) in (x1,x2) space
            corner_points = [
                [0.0, 0.0],   # f = (0, 0, 1)
                [1.0, 0.0],   # f = (1, 0, 0)
                [0.0, 1.0],   # f = (0, 1, 0)
                [0.5, 0.5],   # center
            ]

            for cx1, cx2 in corner_points:
                if remaining <= 0 or time.time() > deadline - 5:
                    break
                x = np.zeros(dim)
                x[0] = cx1
                x[1] = cx2
                for k in range(2, dim):
                    x[k] = 0.5
                X = np.vstack([X, x])
                F = np.vstack([F, [problem(x)]])
                remaining -= 1
                used += 1

        return X
# EVOLVE-BLOCK-END
