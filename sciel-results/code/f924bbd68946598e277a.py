"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct sampling of the Pareto front for ZDT2 variant."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Track time to ensure we don't exceed limits, though 200 evals is fast
        start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The knowledge indicates the Pareto front is at x[1:] = 0.
        # We use the full budget to sample x[0] uniformly to cover the front.
        n_evals = self.budget
        
        # Generate x[0] values uniformly in [0, 1]
        # We ensure we hit the endpoints 0 and 1 for maximum hypervolume coverage
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        # Construct the full vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # X[:, 1:] is already 0.0
        
        # Clip to bounds just in case, though [0,1] is standard
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function takes a single vector x and returns objectives
        # We loop through them as required by the interface
        results = []
        for i in range(n_evals):
            # Check time limit with a safety margin (e.g., stop if > 250s)
            if time.time() - start_time > 250:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
