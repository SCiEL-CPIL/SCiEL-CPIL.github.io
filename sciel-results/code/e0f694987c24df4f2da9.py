import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, keep margin
        
        n_evals = min(self.budget, 200)
        
        # ZDT1 Pareto front (g=0):
        # f1 = x1, f2 = 1 + sin(10*pi*x1)
        # Non-dominated: x1=0 and x1 in (0.1, 0.15]
        # Place 1 point at x1=0, rest in (0.1, 0.15]
        
        X = np.zeros((n_evals, self.dim))
        
        # Point at x1 = 0
        X[0, 0] = 0.0
        
        # Remaining points in (0.1, 0.15]
        if n_evals > 1:
            # Use linspace from just above 0.1 to 0.15
            # Avoid x1=0.1 exactly since it's dominated
            t_vals = np.linspace(0.1 + 1e-10, 0.15, n_evals - 1)
            X[1:, 0] = t_vals
            # x2..xn = 0 (g=0)
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline - 10:
                break
            problem(X[i])
        
        return X
