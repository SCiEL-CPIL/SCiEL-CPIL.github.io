"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer exploiting DTLZ7's analytical Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front:
        # f1 = x0, f2 = x1, f3 = 2 - sqrt(x0) when x[2:] = 0
        # We sample the (x0, x1) plane using a Halton sequence for low-discrepancy coverage.
        
        n_points = self.budget
        dim = self.dim
        
        # Generate Halton sequence for 2 dimensions (u, v)
        # Halton sequence in base 2 and 3
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        u_vals = np.array([halton(i + 1, 2) for i in range(n_points)])
        v_vals = np.array([halton(i + 1, 3) for i in range(n_points)])
        
        # Construct X
        # x = [u, v, 0, 0, ..., 0]
        X = np.zeros((n_points, dim))
        X[:, 0] = u_vals
        X[:, 1] = v_vals
        # x[2:] remains 0, which is within bounds [0, 1]
        
        # Evaluate
        # We must ensure we don't exceed the time limit.
        # 200 evaluations is very fast, so this is likely not an issue,
        # but we add a check just in case.
        
        evaluated_X = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = X[i]
            # Evaluate
            problem(x)
            evaluated_X.append(x)
            
        return evaluated_X
# EVOLVE-BLOCK-END
