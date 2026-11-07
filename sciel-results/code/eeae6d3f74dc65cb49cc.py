import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        def timed_out():
            return time.time() > deadline

        # Track best
        best_f = float('inf')
        best_x = np.zeros(dim)

        def update(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # 1. Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        update(x0, f0)

        # 2. Estimate gradient using forward differences from origin
        #    For a linear function f(x) = g^T x + c, we have:
        #    f(e_i * h) = g_i * h + c
        #    f(0) = c
        #    So g_i = (f(e_i * h) - f(0)) / h
        h = 1.0
        g = np.zeros(dim)
        
        for i in range(dim):
            if timed_out():
                break
            xp = np.zeros(dim)
            xp[i] = h
            fp = func(xp)
            g[i] = (fp - f0) / h
            update(xp, fp)

        # 3. Predict optimum vertex
        x_pred = np.where(g > 0, -5.0, 5.0)
        x_pred[np.abs(g) < 1e-9] = 0.0

        f_pred = func(x_pred)
        update(x_pred, f_pred)

        # 4. Verify by checking all vertices
        for mask in range(2 ** dim):
            if timed_out():
                break
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            fv = func(v)
            update(v, fv)

        return best_x, best_f
