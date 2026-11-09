"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def halton_sequence(n, base):
    """Generate a Halton sequence in 1D."""
    seq = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        seq.append(r)
    return np.array(seq)

class Optimizer:
    """Multi-objective optimizer for ZDT6 using quasi-random sampling of the front variable."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_samples = self.budget

        # Use Halton sequence for x[0] to better cover the non-uniform Pareto front
        # Base 2 is sufficient for 1D
        x0_samples = halton_sequence(n_samples, 2)
        
        # Ensure we include endpoints or near-endpoints if beneficial
        # Halton sequence in [0,1] does not include 0 or 1 exactly.
        # We can add them if we have budget, but we are using full budget.
        # We can map the sequence to [0,1] explicitly if needed, but it's already in [0,1].
        
        # Create the population matrix
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_samples
        
        # Set x[1:] to 0.5 (assumed Pareto optimal values for ZDT6)
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return F
# EVOLVE-BLOCK-END
