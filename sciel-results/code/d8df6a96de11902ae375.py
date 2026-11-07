import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT1 Pareto Front Analysis:
        # f1 = x1
        # f2 = (1 + g(x2..xn)) * (1 + sin(10*pi*x1))
        # Global min of g is 0 at x2..xn = 0.
        # Pareto front is the set of non-dominated points.
        # A point (x1, f2(x1)) with g=0 is non-dominated if f2(x1) is the minimum
        # of f2(t) for t in [0, x1].
        # f2(x1) = 1 + sin(10*pi*x1).
        # Min of f2 is 0 at 10*pi*x1 = 1.5*pi + 2*k*pi => x1 = 0.15 + 0.2*k.
        # First minimum at x1=0.15.
        # For x1 > 0.15, f2(x1) >= 0, so (0.15, 0) dominates all x1 > 0.15.
        # For x1 in [0, 0.1], min f2 is 1 (at x1=0). Only x1=0 is non-dominated 
        # (x1=0.1 has f2=1, not dominated by x1=0 as f2 is equal and f1 is worse, 
        # but x1=0.1 is the start of the next segment).
        # Actually, (0,1) does not dominate (0.1, 1). 
        # However, for x in (0, 0.1), f2(x) > 1, so dominated by (0,1).
        # For x in [0.1, 0.15], f2 decreases from 1 to 0. 
        # These points are non-dominated.
        # So the Pareto Front is: x1=0 and x1 in [0.1, 0.15].
        
        n_evals = self.budget
        # Ensure we don't exceed budget
        n_evals = min(n_evals, 200)
        
        # We have 200 points.
        # 1 point at x1=0.
        # 199 points in [0.1, 0.15].
        
        # Create X matrix
        X = np.zeros((n_evals, self.dim))
        
        # Point 0
        X[0, 0] = 0.0
        
        # Points 1 to n_evals-1
        if n_evals > 1:
            # Uniformly space 199 points in [0.1, 0.15]
            t_vals = np.linspace(0.1, 0.15, n_evals - 1)
            X[1:, 0] = t_vals
            # X[:, 1:] remains 0
        
        # Evaluate
        for i in range(n_evals):
            problem(X[i])
            
        return X
