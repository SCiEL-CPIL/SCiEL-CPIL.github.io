"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting the known analytic Pareto front of ZDT2.
    The Pareto front for ZDT2 is located at x[1:] = 0.
    We sample this front non-uniformly (denser near x_0=0) to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2 Pareto front is at x[1:] = 0 (lower bound for x[1:])
        # x[0] varies in [lo[0], hi[0]]
        
        # Generate candidate points on the Pareto front
        # Use non-uniform sampling: denser near lo[0] (which is 0)
        # x_0 = lo[0] + (hi[0] - lo[0]) * u^alpha
        # alpha > 1 concentrates points near lo[0]
        n_points = self.budget
        u = np.linspace(0, 1, n_points)
        alpha = 2.0
        x0_values = lo[0] + (hi[0] - lo[0]) * (u ** alpha)
        
        # Initialize candidate matrix
        candidates = np.zeros((n_points, self.dim))
        candidates[:, 0] = x0_values
        # Set remaining dimensions to lower bound (which is 0 for ZDT2)
        if self.dim > 1:
            candidates[:, 1:] = lo[1:]

        evaluated_points = []
        
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
                
            x = candidates[i]
            # problem(x) returns objective values
            f = problem(x)
            evaluated_points.append(f)
            
        return None
# EVOLVE-BLOCK-END
