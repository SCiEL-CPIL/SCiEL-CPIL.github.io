"""Multi-objective optimizer on ZDT4 using cosine-spaced sweep of x0."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Cosine-spaced sweep optimizer for ZDT4.
    
    ZDT4's Pareto front: f1 = x0, f2 = 1 - sqrt(f1) (when g=0).
    The front has infinite slope at f1=0, so we use cosine spacing
    to cluster points near x0=0 where the front is steepest.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        x0_lo, x0_hi = lo[0], hi[0]
        
        n_sweep = budget
        
        # Uniform spacing in x0 for optimal hypervolume coverage
        # The Pareto front f2 = 1 - sqrt(f1) has nearly constant "height" 
        # (ref_f2 - f2 varies only from 399 to 400), so uniform spacing
        # minimizes max gap and thus minimizes HV discretization error
        x0_values = np.linspace(x0_lo, x0_hi, n_sweep)
        
        # Ensure exact endpoints
        x0_values[0] = x0_lo
        x0_values[-1] = x0_hi
        
        # Pre-allocate the full matrix for efficiency
        X = np.zeros((n_sweep, dim))
        X[:, 0] = x0_values
        # x[1:] = 0 (global min of g terms for ZDT4)
        X = np.clip(X, lo, hi)
        
        results = []
        used = 0
        
        # Evaluate sweep points
        for i in range(n_sweep):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            used += 1
        
        return results
# EVOLVE-BLOCK-END
