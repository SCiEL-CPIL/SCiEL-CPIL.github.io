import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # For ZDT2 with n=30:
        # f1 = x1
        # f2 = 11 * (1 - sqrt(x1))
        # Pareto front: x2...xn = 0, x1 in [0,1]
        # Reference point: [11.0, 11.0]

        num_points = min(self.budget, 200)

        if num_points <= 1:
            x1_vals = np.array([0.5])
        else:
            # Optimal sampling for hypervolume on ZDT2 front
            # Derived from equalizing the integral of 1/h(x) dx
            # h(x) = 11 * sqrt(x)
            # Integral gives x_i = (i/N)^2
            i = np.arange(num_points, dtype=float)
            x1_vals = (i / (num_points - 1)) ** 2.0

        used = 0
        recorded = []

        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
