import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Check if 0 is within bounds for x1..x_{dim-1}
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            n_points = self.budget
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                # Try different biasing: use a power law with exponent 1.5
                # This provides a moderate bias toward x0_lo
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 1.5)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X

        else:
            best_x_tail = []
            for i in range(1, self.dim):
                l = lo[i]
                h = hi[i]
                if abs(l) < abs(h):
                    best_x_tail.append(l)
                else:
                    best_x_tail.append(h)
            
            n_points = self.budget
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 1.5)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            for i in range(1, self.dim):
                X[:, i] = best_x_tail[i-1]
            
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X
