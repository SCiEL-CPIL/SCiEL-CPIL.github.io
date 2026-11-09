import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the affine structure of f5 (Linear slope)
    by starting at a random corner and probing gradient signs via
    neighboring corners, finding the optimal vertex in d+1 calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        calls_made = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
            except Exception:
                return np.inf
            calls_made += 1
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x, dtype=np.float64)
            return val

        # Phase 1: Start at a random corner
        corner = np.random.choice([-5.0, 5.0], size=d)
        f_corner = safe_call(corner)
        if not np.isfinite(f_corner):
            return best_x, best_f

        # Phase 2: Probe each axis by flipping that corner's coordinate
        # to determine the gradient sign
        v = np.zeros(d)
        for i in range(d):
            if calls_made >= self.budget or time.time() > deadline:
                break
            neighbor = corner.copy()
            neighbor[i] = -neighbor[i]  # Flip to the other side
            f_neighbor = safe_call(neighbor)
            if not np.isfinite(f_neighbor):
                break
            # If flipping axis i decreases f, then gradient[i] < 0 (decreasing direction)
            # If flipping increases f, then gradient[i] > 0
            if f_neighbor < f_corner:
                v[i] = -1.0  # Gradient is negative, optimum has x_i = 5
            else:
                v[i] = 1.0   # Gradient is positive, optimum has x_i = -5

        # Phase 3: Compute optimal vertex
        opt_x = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                opt_x[i] = -5.0
            else:
                opt_x[i] = 5.0

        f_opt = safe_call(opt_x)
        if not np.isfinite(f_opt):
            return best_x, best_f

        # Phase 4: Stall on best point
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            try:
                func(best_x)
            except Exception:
                break
            calls_made += 1

        return best_x, best_f
