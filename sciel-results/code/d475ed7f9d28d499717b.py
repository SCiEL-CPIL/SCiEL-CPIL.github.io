"""Deterministic LHS-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Latin Hypercube Sampling based optimizer for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate LHS samples using multiple seeds for diversity
        # Split budget across 4 LHS runs with different seeds
        n_per_seed = n // 4
        seeds = [42, 123, 456, 789]
        
        all_samples = []
        for seed in seeds:
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            samples = sampler.random(n=n_per_seed)
            all_samples.append(samples)
        
        # If n not divisible by 4, add remaining points from first seed
        remainder = n - n_per_seed * 4
        if remainder > 0:
            sampler_extra = qmc.LatinHypercube(d=dim, seed=999)
            extra = sampler_extra.random(n=remainder)
            all_samples.append(extra)
        
        samples_unit = np.vstack(all_samples)
        
        # Map to bounds
        X = lo + (hi - lo) * samples_unit

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
