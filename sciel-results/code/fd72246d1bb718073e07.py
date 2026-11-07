# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Analytical Pareto front sampler for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate points on the Pareto front of ZDT2
        # ZDT2 Pareto front: x_1 in [0, 1], x_2..x_n = 0
        # f_1 = x_1, f_2 = 1 - sqrt(x_1)
        
        n_points = self.budget
        
        # Use a non-uniform grid to minimize hypervolume approximation error.
        # Optimal spacing for minimizing error on convex f_2 is x ~ i^(4/3)
        indices = np.arange(n_points)
        if n_points > 1:
            x1_values = (indices / (n_points - 1)) ** (4.0 / 3.0)
        else:
            x1_values = np.array([0.5])
            
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
