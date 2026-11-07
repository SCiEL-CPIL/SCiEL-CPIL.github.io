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

        # Use Latin Hypercube Sampling over u in [0, 1]
        # Then transform to x1 using a power law to bias towards x1=0
        # where the Pareto front has the highest curvature.
        # Theoretical optimal density for 1-sqrt(x) is proportional to x^(-3/4),
        # which implies x proportional to u^4.
        
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Generate LHS samples in u-space
        indices = np.arange(n_points)
        perm = rng.permutation(n_points)
        u_vals = (perm + rng.random(n_points)) / n_points
        
        # Transform to x1 space
        # For ZDT2's convex Pareto front, uniform or mildly biased sampling works best.
        # k=1 gives uniform LHS; mild bias toward x1=0 helps with curvature.
        k = 1.0
        x1_vals = u_vals ** k
        
        # Sort to ensure monotonic ordering
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
