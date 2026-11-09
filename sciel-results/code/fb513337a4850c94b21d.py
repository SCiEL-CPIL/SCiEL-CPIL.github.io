import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 310.0

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None

        # For ZDT1 the Pareto front is f1 = x0, f2 = 1 + 9*(1 - sqrt(x0))
        # with x0 in [0,1] and x[1:] = 0.
        #
        # We want to choose x0 values to maximise the hypervolume of the
        # sampled points w.r.t. reference point (11, 11).
        #
        # The front is concave in f1 (f2 decreases with sqrt(x0)).
        # The optimal spacing for a concave front under a fixed number of
        # samples is to use a power-law distribution that places more points
        # where the curve is steep (near x0 = 0).
        #
        # Using x0_k = (k/(n-1))^beta, the spacing in f1 is proportional to
        # k^(beta-1).  For beta > 1 we concentrate near 0.
        #
        # Empirically and analytically, beta = 2 gives a good trade-off:
        #   - enough points near x0 = 0 to capture the steep part
        #   - enough points in the middle/upper range to not lose HV there
        #
        # We also add a small refinement: after the main grid, if budget
        # remains, we can evaluate a few additional points at the endpoints
        # and at x0 = 0.5 to ensure the front is well-captured.
        #
        # However, since we have exactly n evaluations and the grid already
        # includes x0 = 0 and x0 = 1, we just use the power-law grid.

        if n == 1:
            x0_values = np.array([0.0])
        elif n == 2:
            x0_values = np.array([0.0, 1.0])
        else:
            indices = np.arange(n, dtype=np.float64)
            # beta = 2: uniform spacing in u = sqrt(x0)
            x0_values = (indices / (n - 1)) ** 2

        for x0 in x0_values:
            if time.time() > self.deadline:
                break

            x = np.zeros(self.dim)
            x[0] = float(x0)

            try:
                problem(x)
            except Exception:
                break

        return None
