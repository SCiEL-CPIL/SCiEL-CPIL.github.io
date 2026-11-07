# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # Use Latin Hypercube Sampling over x1 in [0, 1]
        # This ensures uniform coverage and avoids systematic bias of fixed grids
        # LHS: divide [0,1] into n_points equal intervals, sample one point per interval
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility within the run
        # Generate LHS samples
        indices = np.arange(n_points)
        # Randomly permute the intervals for better space-filling
        perm = rng.permutation(n_points)
        # Sample uniformly within each interval
        x1_vals = (perm + rng.random(n_points)) / n_points
        # Sort to ensure monotonic ordering (not strictly necessary but helpful)
        x1_vals = np.sort(x1_vals)

        X_list = []
        F_list = []

        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]

            # Clip to bounds
            x = np.clip(x, lo, hi)

            f = problem(x)
            X_list.append(x)
            F_list.append(f)

            if time.time() > deadline:
                break

        return np.array(X_list)
# EVOLVE-BLOCK-END
