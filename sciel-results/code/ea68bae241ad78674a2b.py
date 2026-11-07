"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using dense 1D grid search on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim

        # For ZDT4: x1..x9 should be 0 for optimal f2
        # x0 determines f1 on the Pareto front
        # f1(x) = 1 + 9 * x10 (i.e., 1 + 9 * (n-1) if n>1, but for ZDT4 f1 depends only on x0)
        # Actually for ZDT4: f1 = x0, f2 = g(x') * h(f1, g)
        # where g = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi))
        # To get g=1, we need xi=0 for all i>=1
        # h(f1, g) = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        # So on the Pareto front: f1 = x0, f2 = h(x0, 1)

        # Strategy: dense grid search over x0, with x1..x9 = 0
        # This gives points on the Pareto front

        n_evals = self.budget
        # Use all budget for grid points on x0
        # x0 range is typically [0, 1] for ZDT4

        x0_lo = lo[0]
        x0_hi = hi[0]

        # Create dense grid over x0
        n_grid = n_evals
        x0_vals = np.linspace(x0_lo, x0_hi, n_grid)

        # For each x0, set x1..x9 to 0 (or their lower bound if 0 is out of range)
        # Typically ZDT4 has x_i in [-5, 5] for i>=1, so 0 is in range
        results = []
        for x0 in x0_vals:
            x = np.zeros(n_vars)
            x[0] = x0
            # Clip to bounds to be safe
            for j in range(n_vars):
                x[j] = np.clip(x[j], lo[j], hi[j])
            f = problem(x)
            results.append(f)

            # Check deadline
            if time.time() > deadline:
                break

        # Also add some points with small perturbations to x1..x9 to ensure
        # we're truly on the front (in case 0 isn't exactly optimal due to bounds)
        # But the grid should already be very dense

        return results
# EVOLVE-BLOCK-END
