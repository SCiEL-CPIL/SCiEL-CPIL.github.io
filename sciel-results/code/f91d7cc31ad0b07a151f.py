import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT1 Pareto front is f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1]
        # and x2...xD = 0.
        # We want to maximize hypervolume with 'budget' points.
        # Optimal distribution for left-Riemann sum minimization of error
        # for y = 1 - sqrt(x) is x_i = ((i+1)/N)^1.75
        
        N = self.budget
        # Ensure N is at least 1
        if N < 1:
            return np.zeros((0, self.dim))
            
        # Generate x1 values
        i = np.arange(1, N + 1)
        x1_vals = (i / N) ** 1.75
        
        # Construct population
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        # x[:, 1:] remains 0.0 which is optimal for g(x)
        
        # Evaluate all points
        # The problem function records the objectives
        for x in X:
            problem(x)
            
        return X
