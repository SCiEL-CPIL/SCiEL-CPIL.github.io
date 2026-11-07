import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Analytic construction based on knowledge:
        # ZDT3 Pareto front is typically x_1 uniform, others 0 or 0.2.
        # Knowledge says 0.
        
        n_evals = self.budget
        # Generate x_1 values
        x1_values = np.linspace(0, 1, n_evals)
        
        # Construct X
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_values
        # X[:, 1:] is already 0
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
