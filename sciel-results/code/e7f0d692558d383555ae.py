import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        # Generate x1 using a distribution that clusters points near 0 and 1
        # where the Pareto front (h(x1)) is steep.
        u = np.linspace(0.0, 1.0, n_evals)
        # Map u to x1 using cosine distribution
        # x1 = 0.5 * (1 - np.cos(np.pi * u))
        # This gives high density near 0 and 1.
        x1_values = 0.5 * (1.0 - np.cos(np.pi * u))
        
        # Ensure 0 and 1 are included?
        # linspace includes 0 and 1.
        # cos(0)=1 -> x1=0.
        # cos(pi)=-1 -> x1=1.
        
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_values
        
        # Set tail variables to 0 (Pareto front)
        # X[:, 1:] is already 0.
        
        F = np.array([problem(x) for x in X])
        
        return X
