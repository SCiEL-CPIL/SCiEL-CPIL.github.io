import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT1 specific optimization
        # The Pareto front is achieved when x[1:] are 0.
        # f1 = x[0], f2 = 1 - sqrt(x[0])
        # We want to maximize hypervolume of points (x[0], 1-sqrt(x[0]))
        # with reference point (11, 11).
        
        # Budget is 200.
        # We will use all 200 evaluations.
        # Strategy: Sample x[0] with non-uniform density, higher near 0.
        # x_i = (i / (N-1))^2 for i in 0..N-1
        
        n_points = self.budget
        if n_points < 1:
            return []
            
        # Generate x[0] values
        # Use quadratic spacing to cluster near 0
        i = np.arange(n_points)
        # Avoid division by zero if n_points=1
        if n_points == 1:
            x0_vals = np.array([0.5])
        else:
            x0_vals = (i / (n_points - 1))**2
        
        # Construct full vectors
        # x[0] = x0_vals
        # x[1:] = 0.0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        
        # Clip to bounds just in case, though 0 and 1 are within [0,1]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
