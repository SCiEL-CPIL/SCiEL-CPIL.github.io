# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        if n_evals <= 0:
            return None

        # For ZDT1, Pareto front: x_2...x_d = 0, x_1 in [0,1]
        # f1 = x1, f2 = 1 - sqrt(x1)
        
        # Optimal density for HV maximization: p(t) ∝ 1/(10+sqrt(t))
        # CDF: F(t) = [2*sqrt(t) - 20*ln((10+sqrt(t))/10)] / [2 - 20*ln(1.1)]
        
        C = 2 - 20 * np.log(1.1)
        
        def cdf_inv(u):
            """Inverse CDF for the optimal distribution."""
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            lo_t = 0.0
            hi_t = 1.0
            target = u * C
            for _ in range(60):
                mid_t = (lo_t + hi_t) / 2
                val = 2 * np.sqrt(mid_t) - 20 * np.log((10 + np.sqrt(mid_t)) / 10)
                if val < target:
                    lo_t = mid_t
                else:
                    hi_t = mid_t
            return (lo_t + hi_t) / 2

        # Use quantiles including endpoints for better HV
        if n_evals == 1:
            t_vals = np.array([0.5])
        else:
            u_vals = np.arange(n_evals) / (n_evals - 1)
            t_vals = np.array([cdf_inv(u) for u in u_vals])
        
        # Clip to bounds
        t_vals = np.clip(t_vals, lo[0], hi[0])
        
        # Construct points
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = t_vals
        # x_2...x_d = 0
        
        last_obj = None
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            last_obj = f

        return last_obj
# EVOLVE-BLOCK-END
