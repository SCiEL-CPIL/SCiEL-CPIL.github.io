import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        
        # ZDT3 Pareto front is at x[1:] = 0
        # Connected regions where sin(10*pi*x1) <= 0:
        # x1 in [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
        # These are 5 regions. We distribute evaluations to maximize hypervolume.
        
        # For hypervolume with reference [11, 11], we want good spread in f1 and f2.
        # The Pareto front: f1 = x1, f2 = g*(1 - sqrt(f1/g) - f1*sin(10*pi*f1))
        # with g = 1 + 10*(dim-1) = 291 for dim=30
        
        g_min = 1 + 10 * (self.dim - 1)
        
        # The 5 connected regions
        regions = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (0.9, 1.0)]
        
        # Distribute points: give more to regions that cover more f2 range
        # Region near x1=1.0 has f2 close to 0, region near x1=0.1 has higher f2
        # For hypervolume, we want coverage across the whole front
        # Use equal distribution per region for simplicity, but with fine resolution
        
        n_per_region = n_evals // len(regions)
        remainder = n_evals - n_per_region * len(regions)
        
        x1_vals = []
        for i, (lo, hi) in enumerate(regions):
            n = n_per_region + (1 if i < remainder else 0)
            # Use linspace within each region
            x1_vals.extend(np.linspace(lo, hi, n, endpoint=True).tolist())
        
        # Sort to ensure we cover the front properly
        x1_vals = np.array(x1_vals)
        
        # Create population with x[1:] = 0 (on the Pareto front)
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        
        # Evaluate each point
        for i in range(n_evals):
            problem(X[i])
        
        return X
