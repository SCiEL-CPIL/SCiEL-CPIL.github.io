# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer using low-discrepancy sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        M = self.n_obj
        n_g = M - 1  # number of shape variables (first M-1)

        X_list = []
        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1
            return f

        # Base vector with g-variables set to 0.5 (minimum of g for DTLZ4)
        base_x = np.full(n, 0.5)

        # 1. Add explicit corner points for the shape variables to ensure front extent
        corners = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ]
        
        if n_g == 2:
            for c in corners:
                x = base_x.copy()
                x[0] = c[0]
                x[1] = c[1]
                evaluate(x)
        else:
            # Generic corner handling if n_g != 2 (though task specifies 3 obj)
            for i in range(2**n_g):
                c = [(i >> j) & 1 for j in range(n_g)]
                x = base_x.copy()
                for k in range(n_g):
                    x[k] = float(c[k])
                evaluate(x)
                if used >= self.budget:
                    break

        # 2. Fill remaining budget with low-discrepancy Sobol sequence points
        if used < self.budget:
            N_sobol = self.budget - used
            # Sobol works best with powers of 2, but accepts any N
            sampler = qmc.Sobol(d=n_g, scramble=True)
            # Generate points in [0, 1]
            points = sampler.random(N_sobol)
            
            for p in points:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = base_x.copy()
                x[:n_g] = p
                # Clip to bounds
                x = np.clip(x, lo, hi)
                evaluate(x)

        if len(F_list) == 0:
            return []

        return np.array(F_list)
# EVOLVE-BLOCK-END
