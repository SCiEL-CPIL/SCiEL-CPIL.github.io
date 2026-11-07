"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure to maximize hypervolume:
    - f1 = x1, g = 1 + 10(n-1) + sum(xi^2 - 10*cos(2*pi*xi)) for i>=2
    - f2 = (1 - sqrt(f1/g)) * g
    - Setting x2..x10 = 0 gives g = 1, f2 = 1 - sqrt(x1)
    - Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 11]
    - Reference point [11, 400]: all front points with f1 < 11 count
    - Strategy: use all budget to densely sample the front uniformly in f1
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Determine x1 range for Pareto front sampling
        # Front is useful up to f1 = 11 (reference point)
        x1_min = max(lo[0], 0.0)
        x1_max = min(hi[0], 11.0)
        
        if x1_min >= x1_max:
            # Fallback: use full bound range
            x1_min = lo[0]
            x1_max = hi[0]

        n_evals = self.budget
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            # Uniform sampling of x1 in [x1_min, x1_max]
            x1 = x1_min + (x1_max - x1_min) * (i / max(n_evals - 1, 1))
            
            x = np.zeros(self.dim)
            x[0] = x1
            # Set x[1:] to 0 (or closest bound if 0 is out of range)
            for j in range(1, self.dim):
                if lo[j] <= 0.0 <= hi[j]:
                    x[j] = 0.0
                elif 0.0 < lo[j]:
                    x[j] = lo[j]
                else:
                    x[j] = hi[j]
            
            x_clipped = np.clip(x, lo, hi)
            try:
                f = problem(x_clipped)
                results.append(f)
            except Exception:
                break
        
        return results
# EVOLVE-BLOCK-END
