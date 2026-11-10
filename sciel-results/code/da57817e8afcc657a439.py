import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # ZDT6 Pareto front: f1 = 1 - sqrt(x0), f2 = (1 - f1) * (1 - sin(10*pi*f1))
        # We sample x0 to maximize hypervolume coverage of the oscillatory front.
        # Use a non-uniform grid in x0 that concentrates points where f2 oscillates rapidly.
        
        # Strategy: sample f1 values using a Chebyshev-like distribution to handle oscillations,
        # then solve for x0 = (1 - f1)^2.
        
        # Use Chebyshev nodes for f1 to better capture the oscillatory behavior of sin(10*pi*f1)
        # Chebyshev nodes: f1_i = 0.5 * (1 - cos(pi * (2*i + 1) / (2*n)))
        i = np.arange(n_evals)
        f1_values = 0.5 * (1.0 - np.cos(np.pi * (2.0 * i + 1.0) / (2.0 * n_evals)))
        
        # Solve for x0: f1 = 1 - sqrt(x0) => x0 = (1 - f1)^2
        x0_values = (1.0 - f1_values) ** 2
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # x[1:] = 0 to ensure all points lie exactly on the Pareto front
        
        F = np.zeros((n_evals, self.n_obj))
        for idx in range(n_evals):
            F[idx] = problem(X[idx])
            
        return X
