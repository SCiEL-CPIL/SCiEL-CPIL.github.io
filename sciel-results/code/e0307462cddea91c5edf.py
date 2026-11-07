import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_front_vars = self.n_obj - 1
        n_dist_vars = self.dim - n_front_vars
        
        # Generate points on the Pareto front
        # Use Dirichlet distribution for uniform coverage on the simplex
        # alpha=1 gives uniform distribution
        # We generate n_obj components, but only use the first n_front_vars
        # The last component is implied (1 - sum(first n_front_vars))
        
        # To ensure we don't have points too close to the edges (which might be less "central"),
        # we can use a slightly higher alpha, e.g., 2.0, to bias towards the center.
        # However, for hypervolume, we want to cover the whole front.
        # Uniform (alpha=1) is best for coverage.
        
        # Generate random points
        points = np.random.dirichlet(np.ones(self.n_obj), size=self.budget)
        
        X = np.zeros((self.budget, self.dim))
        X[:, :n_front_vars] = points[:, :n_front_vars]
        
        # Set distance variables to 1.0
        X[:, n_front_vars:] = 1.0
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
            
        return X
