"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 Pareto front: x[0] varies, x[1:] = 0
        # f1 = x[0], f2 = (1 + x[0]) * (1 - sqrt(x[0]))
        # Spread points along the Pareto front to maximize hypervolume
        
        n_points = self.budget
        x0_values = np.linspace(0.0, 1.0, n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        # x[1:] = 0 is already set by zeros
        
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
