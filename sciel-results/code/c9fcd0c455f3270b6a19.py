"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search focused on the guaranteed minimizer region.
    
    Strategy:
    1. The problem states the minimizer lies in [-4, 4]^5.
    2. We use a scrambled Sobol sequence (deterministic via fixed seed) for space-filling properties.
    3. We map the points to [-4, 4]^5 to maximize density in the region of interest.
    4. We evaluate the first 500 points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        from scipy.stats import qmc
        
        # Generate Sobol points
        # Use scramble=True with a fixed seed for deterministic but well-distributed points
        # that avoid systematic alignment with potential hidden rotations
        n_sobol = 512  # Next power of 2 >= 500
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit = sobol.random(n_sobol)
        
        # Map from [0, 1] to [-4, 4]
        # The minimizer is guaranteed to be in [-4, 4]^5, so we focus our budget here
        # This increases the effective resolution by (10/8)^5 ~ 3x compared to [-5, 5]
        points = points_unit * 8.0 - 4.0
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        for i in range(min(n_sobol, budget)):
            x = points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
