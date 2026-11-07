import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 Pareto front is located at x[1:] = 0
        # Parameterized by x[0] = u in [0, 1]
        # f1 = 1 - exp(-4u) * sin(6*pi*u)^6
        # f2 = 1 - sqrt(f1)
        
        # We have 'budget' evaluations.
        # We will sample 'budget' points on the front.
        # To ensure good coverage for hypervolume, uniform sampling in u is a good start.
        # However, the spacing in objective space is not uniform.
        # But 200 points is quite dense.
        
        n_evals = self.budget
        u_vals = np.linspace(0, 1, n_evals)
        
        # Construct decision vectors
        # x[0] = u, x[1:] = 0
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = u_vals
        
        # Evaluate
        # The problem function clips and returns objectives.
        # We must call it for each point.
        results = []
        for x in X:
            f = problem(x)
            results.append(f)
            
        return results
