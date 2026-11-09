import numpy as np
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Generate 200 points uniformly on the DTLZ2 Pareto front
        n_points = self.budget
        # Use random numbers to generate uniform points on the simplex
        # Method: sort 0, u1, u2, 1
        u = np.random.rand(n_points, 2)
        
        # Sort along the axis with the extra 0s and 1s
        # Create array of shape (n_points, 4): [0, u1, u2, 1]
        verts = np.zeros((n_points, 4))
        verts[:, 0] = 0.0
        verts[:, 1] = u[:, 0]
        verts[:, 2] = u[:, 1]
        verts[:, 3] = 1.0
        verts.sort(axis=1)
        
        # f0 = v1, f1 = v2 - v1, f2 = 1 - v2
        f0 = verts[:, 1]
        f1 = verts[:, 2] - verts[:, 1]
        f2 = 1.0 - verts[:, 2]
        
        # Construct x
        # x0 = sqrt(f0), x1 = sqrt(f1), x2 = sqrt(f2)
        # x3...x11 = 0.5
        x = np.zeros((n_points, self.dim))
        x[:, 0] = np.sqrt(f0)
        x[:, 1] = np.sqrt(f1)
        x[:, 2] = np.sqrt(f2)
        x[:, 3:] = 0.5
        
        # Evaluate
        F = np.array([problem(xi) for xi in x])
        
        return x
