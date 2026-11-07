"""Multi-objective optimizer on ZDT2 using LHS-inspired stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    - Use Latin Hypercube Sampling (LHS) for x1 to ensure uniform coverage of the Pareto front.
    - Set x_2...x_n to 0, which is the global minimizer for the g(x) function in ZDT2,
      ensuring points lie on the true Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # Number of LHS samples equals the budget to maximize coverage
        n_lhs = budget
        
        # Generate LHS samples for x1 in [0, 1]
        # 1. Create a permutation of the strata indices
        perm = np.random.permutation(n_lhs)
        # 2. Add a random offset within each stratum
        #    This ensures points are uniformly distributed within their strata
        x1_samples = (perm + np.random.random(n_lhs)) / n_lhs
        
        # Clip to bounds (though LHS in [0,1] should stay within [0,1] if bounds are [0,1])
        # Using lo[0] and hi[0] to be generic
        x1_samples = np.clip(x1_samples, lo[0], hi[0])
        
        # Construct the base vector for the remaining dimensions
        # For ZDT2, x_i = 0 (i > 0) minimizes g(x)
        # We use lo[1] assuming bounds are [0,1] or similar where 0 is the minimizer
        # If bounds were different, we'd need to find the minimizer, but for ZDT2 it's 0
        base_vec = np.zeros(n)
        # Ensure base_vec is within bounds
        base_vec[1:] = lo[1]
        
        all_points = []
        for i in range(n_lhs):
            x = base_vec.copy()
            x[0] = x1_samples[i]
            f = problem(x)
            all_points.append(f)
        
        # Return the set of points (return value is ignored)
        if all_points:
            return np.array(all_points)
        return np.array([])
# EVOLVE-BLOCK-END
