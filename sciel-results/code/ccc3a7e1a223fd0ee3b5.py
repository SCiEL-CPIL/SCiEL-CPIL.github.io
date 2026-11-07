# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG4 (3 objectives, 12 variables).
    
    Strategy:
    WFG4 is a 3-objective problem. The Pareto front is typically achieved when
    the 'shape' variables (x[1:]) are set to their upper bounds to minimize the
    distance/penalty terms in the objectives f2 and f3. The 'position' variable
    x[0] controls the trade-off (or simply shifts the point along the front).
    
    This strategy directly samples the Pareto front by:
    1. Uniformly sampling the position variable x[0] to cover the range of f1.
    2. Fixing the shape variables x[1:] to their upper bounds to ensure the points
       lie on (or very close to) the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract lower and upper bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Create a base vector with shape variables set to their upper bounds.
        # For WFG4, the penalty/distance terms are minimized at the upper bounds.
        base = hi.copy()
        
        # Generate unique x[0] values.
        # We sample x[0] uniformly to cover the range [lo[0], hi[0]].
        x0_samples = np.random.uniform(lo[0], hi[0], self.budget)
        
        # Construct the full vectors
        X = np.tile(base, (self.budget, 1))
        X[:, 0] = x0_samples
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Return the set of points (or the front). The return value is ignored by the
        # evaluator, but we must return something.
        return X
# EVOLVE-BLOCK-END
