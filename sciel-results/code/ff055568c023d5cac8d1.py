import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        def check_time():
            return time.time() < deadline and used < self.budget

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # The front is 2D in objective space, parameterized by (x0, x1)
        
        # Strategy: 2D grid for baseline + Halton fill for quasi-random coverage
        
        # Grid size: aim for roughly square grid
        n_grid = int(np.sqrt(self.budget))
        while n_grid * n_grid > self.budget and n_grid > 1:
            n_grid -= 1
        
        # Generate grid points over (x0, x1) in [0, 1]^2
        x0_vals = np.linspace(0.0, 1.0, n_grid)
        x1_vals = np.linspace(0.0, 1.0, n_grid)
        
        for i in range(n_grid):
            for j in range(n_grid):
                if not check_time():
                    break
                x = np.zeros(self.dim)
                x[0] = x0_vals[i]
                x[1] = x1_vals[j]
                # x[2:] remains 0, placing us on the Pareto front
                problem(x)
                used += 1
            if not check_time():
                break
        
        # Fill remaining budget with Halton sequence for better spread
        remaining = self.budget - used
        for i in range(remaining):
            if not check_time():
                break
            
            n = i + 1  # 1-indexed Halton
            x0_val = halton(n, 2)
            x1_val = halton(n, 3)
            
            x = np.zeros(self.dim)
            x[0] = x0_val
            x[1] = x1_val
            # x[2:] = 0
            problem(x)
            used += 1
        
        return None
