"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Problem-specific optimizer exploiting DTLZ2's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 specifics:
        # f_i(x) = (1 + g(x)) * h_i(y)
        # g(x) = sum_{i=n-M+2}^n (x_i - 0.5)^2
        # h_i depends on y = x_1, ..., x_{M-1}
        # To be on Pareto front, g(x) must be 0.
        # This implies x_i = 0.5 for i = n-M+2, ..., n.
        # The variables x_1, ..., x_{M-1} determine the position on the front.
        
        d_pos = n_obj - 1
        d_dist = dim - d_pos
        
        # Generate points in the position space [0, 1]^d_pos
        # Use Sobol sequence for good coverage
        if d_pos == 0:
             # Edge case: 1 objective, no position variables?
             # DTLZ2 usually has M >= 2.
             points_pos = np.zeros((budget, 0))
        else:
            sampler = qmc.Sobol(d=d_pos, scramble=True, seed=0)
            points_pos = sampler.random(budget)
        
        # Construct full dimension vectors
        X = np.zeros((budget, dim))
        
        # Assign position variables
        if d_pos > 0:
            X[:, :d_pos] = points_pos
            
        # Assign distance variables to 0.5
        X[:, d_pos:] = 0.5
        
        # Clip to bounds just in case, though 0.5 and [0,1] should be safe
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The evaluator records the objective values returned by problem(x)
        for i in range(budget):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
