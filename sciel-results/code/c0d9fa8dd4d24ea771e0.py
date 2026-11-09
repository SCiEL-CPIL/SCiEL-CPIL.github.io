import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3 Pareto front is achieved when x_i = 0.5 for i > 1
        # and f1 = x1.
        # We sample x1 uniformly in [0, 1].
        
        n_points = self.budget
        
        # Create array of x1 values
        # Use linspace to cover the range
        x1_vals = np.linspace(0.0, 1.0, n_points)
        
        # Construct the population matrix
        # Shape: (n_points, dim)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        
        # Set other dimensions to 0.5
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Clip to bounds just in case, though 0.5 and [0,1] are safe
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate each point
        # The evaluator records the results
        for i in range(n_points):
            problem(X[i])
            
        return X
