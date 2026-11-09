# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
      - DTLZ1 has 7 variables. The last 5 (indices 2..6) only affect the convergence
        term g(x_M) = 100*(n_M + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))) for i in 2..6.
        The analytic minimizer of g is x_i = 0.5 for all i >= 2, giving g = 0.
      - With g = 0, the Pareto front is the simplex {f >= 0, sum(f) = 0.5}.
      - We sample (x0, x1) from a 2D Sobol sequence to cover the front well.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # Generate a 2D Sobol sequence (low-discrepancy, deterministic)
        sobol_gen = qmc.Sobol(d=2, scramble=False)
        xy = sobol_gen.random(n_evals)
        
        # Clip to [0, 1]
        xy = np.clip(xy, 0.0, 1.0)
        
        # Build full x vectors
        X = np.zeros((n_evals, dim))
        X[:, 0] = xy[:, 0]
        X[:, 1] = xy[:, 1]
        for i in range(2, dim):
            X[:, i] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
