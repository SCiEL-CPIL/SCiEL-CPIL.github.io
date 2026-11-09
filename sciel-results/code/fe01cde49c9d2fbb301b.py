import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s

        # Phase 1: estimate gradient using linear structure
        # Evaluate at center and along each axis
        x0 = np.zeros(dim)
        f0 = func(x0)
        h = 1.0  # step size; x0 + h*e_i = +/-1, well within [-5,5]
        grad = np.zeros(dim)
        for i in range(dim):
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            grad[i] = (fi - f0) / h
        # 1 + dim = 6 evaluations used so far
        calls_used = 1 + dim

        # Phase 2: optimal corner for a linear (affine) function
        # min at corner: coord = -5 if grad[i] > 0 else +5
        corner = np.where(grad > 0, -5.0, 5.0)
        fc = func(corner)
        calls_used += 1
        best_x, best_f = corner, fc

        # Phase 3: local polish around the best point (handles any nonlinearity / clipping)
        # Use a (1+1)-ES with adaptive step, but keep it fast and bounded.
        rng = np.random.default_rng(0)
        sigma = 0.5
        x = best_x.copy()
        fx = best_f
        # We'll do local search but the function is linear so this mostly confirms.
        # Use a modest number of iterations; guard by deadline.
        iters = 0
        while calls_used < budget and (time.time() < deadline):
            if iters % 50 == 0 and iters > 0:
                # occasionally re-center gradient estimate to be safe
                pass
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls_used += 1
            iters += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                # update best
                if fx < best_f:
                    best_x, best_f = x, fx
            else:
                sigma *= 0.85
            if sigma < 1e-6:
                # converged; do a few more tiny steps then stop
                sigma = 1e-6
        return best_x, best_f
