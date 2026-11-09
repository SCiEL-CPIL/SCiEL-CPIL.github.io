"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Sobol sequence sampling for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Sobol sequence for low-discrepancy sampling
        # We need to ensure we don't exceed budget
        # scipy.stats.qmc.Sobol requires dimensionality power of 2 for best performance, but works for any
        # We will generate points in [0, 1]^dim and map to [lo, hi]^dim
        
        from scipy.stats import qmc
        
        # Generate Sobol sequence
        # start_point is optional, but we can just use the default
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        
        # We need to be careful with the number of points
        # Sobol sequence works best with power of 2, but we can generate exactly 'budget' points
        # Actually, Sobol requires the number of points to be a power of 2 for the standard implementation
        # or we can just generate a large power of 2 and take the first 'budget' points?
        # No, Sobol generates a sequence. We can just ask for 'budget' points?
        # scipy.stats.qmc.Sobol.random(n) generates n points. It handles non-power-of-2 by generating the next power of 2 and slicing?
        # Let's check: "The number of points must be a power of 2." 
        # Wait, the doc says: "n : int, optional. Number of points to generate. Default is 2 ** dim."
        # It does NOT say it must be a power of 2 in recent versions? 
        # Actually, looking at scipy docs: "n : int, optional. Number of points to generate. Default is 2 ** dim. ... The number of points must be a power of 2."
        # So we need to generate a power of 2 >= budget and slice.
        
        next_pow2 = 1
        while next_pow2 < self.budget:
            next_pow2 *= 2
            
        # Generate next_pow2 points
        # This might be slightly more than budget, but we only evaluate 'budget' points
        # We will evaluate the first 'budget' points from the Sobol sequence
        points_unit = sampler.random(next_pow2)
        
        # Map to bounds
        X = lo + (hi - lo) * points_unit[:self.budget]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
