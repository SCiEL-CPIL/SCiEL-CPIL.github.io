"""Deterministic LHS-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Sobol sequence based optimizer for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Use Sobol sequence for better uniformity in high dimensions
        # Sobol requires n to be a power of 2 for best properties, but we can take the first n points.
        # We will generate a Sobol sequence of size 2^k >= n and take the first n points.
        # This ensures a deterministic, well-distributed set of points.
        
        # Find the next power of 2 >= n
        # n=200 -> 256
        k = 1
        while (1 << k) < n:
            k += 1
        n_sobol = 1 << k
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples_unit = sampler.random(n=n_sobol)
        
        # Take the first n points
        samples_unit = samples_unit[:n]
        
        # Map to bounds
        X = lo + (hi - lo) * samples_unit

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
