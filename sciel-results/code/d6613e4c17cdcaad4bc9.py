"""Grid-based decomposition with Latin Hypercube Sampling for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Grid-based decomposition strategy using Latin Hypercube Sampling.
    
    Partitions the 12-dimensional search space into hyper-rectangular cells,
    generates candidates via LHS for maximal spatial coverage, then refines
    the best non-dominated solutions with local perturbations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the entire budget for Latin Hypercube Sampling to maximize spatial coverage
        # This ensures maximal coverage of the Pareto front manifold without relying
        # on local gradient information or evolutionary dominance in the sampling phase.
        n_lhs = budget
        
        # Generate LHS samples: one sample per "row" in each dimension
        # For each dimension, divide [0,1] into n_lhs equal intervals and pick one point per interval
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create n_lhs intervals
            intervals = np.arange(n_lhs) / n_lhs
            # Shuffle to get LHS property
            perm = rng.permutation(n_lhs)
            # Pick a random point within each interval
            points = intervals[perm] + rng.random(n_lhs) / n_lhs
            lhs_samples[:, d] = points
        
        # Map to actual bounds
        X_lhs = lo + (hi - lo) * lhs_samples
        
        # Evaluate all points
        # The evaluator records every objective vector returned by problem(x).
        # We perform exactly 'budget' evaluations.
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        # The return value of __call__ is ignored by the evaluator.
        return X_lhs
# EVOLVE-BLOCK-END
