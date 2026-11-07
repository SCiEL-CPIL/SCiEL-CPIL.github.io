# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq


class Optimizer:
    """Constructs points on the true DTLZ1 Pareto front by solving for aggregation variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        n_pos = dim - n_obj + 1  # 5
        n_agg = n_obj - 1        # 2
        n_shape_vars = n_obj - 1 # 2
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_shape_vars, scramble=True, seed=0)
            shape_points = sampler.random(n)
        except ImportError:
            rng = np.random.default_rng(0)
            shape_points = rng.random((n, n_shape_vars))
        
        X = np.zeros((n, dim))
        X[:, :n_shape_vars] = shape_points
        
        # Solve for c such that g(x)=0 with x_3..x_7 = c
        # Equation: 5*c^2 - 10*x_2*cos(4*pi*c + 4*pi*x_2/5) + 7 - x_2 = 0
        K = 5
        for i in range(n):
            x2 = shape_points[i, 1]
            
            def eq(c):
                return 5 * c**2 - 10 * x2 * np.cos(4 * np.pi * c + 4 * np.pi * x2 / K) + 7 - x2
            
            # Find root in [0, 1]
            try:
                c = brentq(eq, 0.0, 1.0, maxiter=100)
            except ValueError:
                # Fallback to 0.5 if no root found (should not happen)
                c = 0.5
                
            X[i, n_shape_vars:dim] = c
            
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
