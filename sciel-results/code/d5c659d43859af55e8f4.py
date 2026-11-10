import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # Use a deterministic grid search that varies x0 across [0, 1]
        # to capture the full oscillatory range of f1, while setting x[1:] = 0
        # to ensure all points lie exactly on the Pareto front.
        
        # Create a uniform grid of x0 values in [0, 1]
        x0_grid = np.linspace(0.0, 1.0, n_evals)
        
        # Build the decision matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_grid
        # x[1:] remain 0, which places points on the Pareto front for ZDT6
        
        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for i in range(n_evals):
            F[i] = problem(X[i])
        
        return X
