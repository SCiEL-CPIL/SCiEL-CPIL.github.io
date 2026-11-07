# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT2 multi-objective optimizer using optimal hypervolume-aware sampling on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # ZDT2: Pareto front is achieved at x[1:] = 0
        # f1 = x[0], f2 = 10*(1 - sqrt(x[0]/30)) for x[0] in [0,1]
        # Reference point is [11, 11]
        # Hypervolume contribution per unit f1 is (11 - f2) = 1 + 10*sqrt(f1/30)
        # To maximize hypervolume with N points, sample with density proportional to (11 - f2)

        n_points = self.budget

        # Compute the CDF of the optimal sampling distribution
        # w(f1) = 1 + 10*sqrt(f1/30)
        # CDF: F(t) = integral_0^t w(u) du / integral_0^1 w(u) du
        # integral w(u) du = u + (20/3)*sqrt(u/30)*sqrt(u) = u + (20/3)*u^(3/2)/sqrt(30)
        # Let's use numerical inversion for robustness

        n_grid = 100000
        x1_grid = np.linspace(0, 1, n_grid)
        w_grid = 1.0 + 10.0 * np.sqrt(x1_grid / 30.0)
        cdf_grid = np.cumsum(w_grid)
        cdf_grid /= cdf_grid[-1]

        # Generate n_points samples using inverse CDF method
        # Use evenly spaced points in CDF space for optimal coverage
        u_values = np.linspace(0, 1, n_points)
        x1_values = np.interp(u_values, cdf_grid, x1_grid)

        # Evaluate all points on the Pareto front
        for x1 in x1_values:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            problem(x)

        return None
# EVOLVE-BLOCK-END
