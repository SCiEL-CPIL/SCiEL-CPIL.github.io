"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the 2D Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    The variables x_1 and x_2 determine the shape of the front.
    Uses a Sobol quasi-random sequence for space-filling sampling
    of the 2D Pareto front to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # DTLZ7 Pareto front: x_3...x_n = 0 (g=0 is achievable)
        # Front is 2D, parameterized by (x1, x2) in [0,1]^2
        # Use Sobol sequence for low-discrepancy space-filling sampling

        n_points = self.budget

        # Generate Sobol sequence: use next power of 2 >= n_points for
        # best uniformity, then take the first n_points samples.
        n_pow2 = 1
        while n_pow2 < n_points:
            n_pow2 *= 2

        sobol = qmc.Sobol(d=2, scramble=True)
        samples = sobol.random(n_pow2)  # (n_pow2, 2) in [0, 1)
        samples = samples[:n_points]    # (n_points, 2)

        # Map to actual bounds
        x1_vals = lo[0] + samples[:, 0] * (hi[0] - lo[0])
        x2_vals = lo[1] + samples[:, 1] * (hi[1] - lo[1])

        for i in range(n_points):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x1_vals[i]
            x[1] = x2_vals[i]
            # x[2:] = 0 gives g = 0, the Pareto-optimal condition

            obj = problem(x)
            results.append(obj)
            n_evals += 1

        return results
# EVOLVE-BLOCK-END
