# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start coarse search + adaptive (1+1)-ES with a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0
        dim = self.dim
        rng = np.random.default_rng(0)
        calls = 0

        def feval(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: coarse multi-start to locate the basin (fast AOCC) ----
        # Lattice of 3^5 = 243 points spread across the box.
        grid = np.array(np.meshgrid(*([np.linspace(-5, 5, 3)] * dim)),
                        dtype=float).reshape(-1, dim)
        order = rng.permutation(grid.shape[0])
        x = np.full(dim, np.nan)
        fx = np.inf
        for idx in order:
            x = grid[idx]
            fx = feval(x)
            if calls > self.budget:
                break
            if time.time() > deadline:
                break
        if not np.isfinite(fx):
            x = np.zeros(dim)
            fx = feval(x)

        # ---- Phase 2: adaptive (1+1)-ES refinement with step-size control ----
        # Ill-conditioned landscape: keep a running success-rate (1/5 rule)
        # for the step size, plus a global best to avoid losing progress.
        sigma = 1.5
        best_x, best_fx = x.copy(), fx
        sigma_log = np.log(sigma)
        chi = 1.0
        rho = 10.0
        c_sigma = 0.01 + 2.0 / max(1, dim - 1)
        d_sigma = 1.0 + 0.5 * c_sigma + 2.0 * max(0.0, (dim - 2) / (dim + 2.0))
        steps = 0
        while calls < self.budget and time.time() < deadline:
            steps += 1
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = feval(y)
            success = fy < fx
            if success:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            # step-size update (success-rate rule with damping)
            chi = (1.0 - c_sigma) * chi + c_sigma * (1.0 if success else 0.0)
            sigma_log += (c_sigma / d_sigma) * (chi - 0.5)
            # clamp sigma to a sensible range
            if sigma_log > np.log(10.0):
                sigma_log = np.log(10.0)
            if sigma_log < np.log(1e-4):
                sigma_log = np.log(1e-4)
            sigma = float(np.exp(sigma_log))

        return best_x, best_fx
# EVOLVE-BLOCK-END
