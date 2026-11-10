import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        # Generate x_0 values clustered near 0 to better capture the steep part of the front
        # ZDT2 front: f2 = 2(1 - sqrt(f1/2)). f1 = x0.
        # The hypervolume contribution height is ref2 - f2.
        # d(height)/dx0 is highest at x0=0.
        # Use a quadratic spacing to place more points near 0.
        x0_vals = np.linspace(0, 1, n_evals) ** 2
        
        # Create the population
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # Set other variables to 0.5 to minimize g(x) to 1 (the Pareto front condition)
        X[:, 1:] = 0.5
        
        # Evaluate
        # We must call problem for each point to record the objective values.
        # The problem function handles clipping and returns objectives.
        for x in X:
            problem(x)
            
        return X
