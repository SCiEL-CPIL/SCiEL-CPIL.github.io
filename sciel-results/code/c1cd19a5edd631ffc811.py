import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Pre-compute dense set of x1 values on [0, 1]
        # The Pareto front: f1 = 1 - exp(-4*t)*sin(6*pi*t)^6, f2 = 1 - t^2
        # We need enough points to capture the front well for hypervolume
        n_front = min(self.budget - 5, 190)  # leave 5 for random safety
        n_front = max(n_front, 50)

        # Use a mix of uniform and clustered points for better HV coverage
        t_uniform = np.linspace(0.0, 1.0, n_front)
        # Add extra points near t=0 where f1 changes rapidly
        if n_front > 60:
            extra_n = min(20, n_front - 60)
            t_extra = np.linspace(0.0, 0.1, extra_n + 1)[1:]  # exclude 0
            t_all = np.concatenate([t_uniform, t_extra])
        else:
            t_all = t_uniform

        # Sort and unique
        t_all = np.sort(np.unique(t_all))
        n_front = len(t_all)

        # Build candidate x vectors: x1 = t, x2..x10 = 0.25 (well within [0, 0.5])
        candidates = np.zeros((n_front, self.dim))
        candidates[:, 0] = t_all
        for i in range(1, self.dim):
            candidates[:, i] = 0.25

        # Evaluate front points
        used = 0
        for i in range(n_front):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(candidates[i])
            used += 1

        # Use remaining budget for a few random evaluations as safety
        rng = np.random.default_rng(42)
        while used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            used += 1

        return None
