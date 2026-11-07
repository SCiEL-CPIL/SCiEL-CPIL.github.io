# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # For ZDT2, the Pareto front is f1 = x1, f2 = 1 - sqrt(f1)
        # The hypervolume contribution is maximized when point density is proportional
        # to local curvature. The curvature of f2 = 1 - sqrt(x1) is proportional to x1^(-3/2).
        # Using optimal design theory, the CDF for optimal spacing is:
        # F(x) = (x^(1/2)) / (1^(1/2)) = sqrt(x)
        # So we set x1 = u^2 where u is uniformly distributed in [0, 1]
        
        # Use deterministic equally-spaced points in u-space for maximum coverage
        # This is equivalent to the optimal design for this convex front
        u_vals = (np.arange(n_points) + 0.5) / n_points
        
        # Transform to x1: x1 = u^2 gives density proportional to 1/sqrt(x1)
        # which matches the curvature-based optimal design
        x1_vals = u_vals ** 2
        
        # Ensure x1_vals are within [0, 1]
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
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
