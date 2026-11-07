"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Latin Hypercube Sampling for broad exploration."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard: 330s wall-clock limit.
        # 200 evaluations of WFG9 is extremely fast, but we check just in case.
        start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = self.budget
        dim = self.dim
        
        # Generate Latin Hypercube Sampling (LHS) points
        # This ensures that each dimension is covered uniformly.
        # For each dimension d, we divide the range into n_samples bins.
        # We assign each sample to a unique bin in that dimension.
        
        lhs = np.zeros((n_samples, dim))
        
        # To avoid overhead of a loop for large dim, we can vectorize, 
        # but dim=12 is small. A loop is clear and safe.
        for d in range(dim):
            # Permutation of indices 0 to n_samples-1
            perm = np.random.permutation(n_samples)
            # Uniform random offset within the bin
            offsets = np.random.rand(n_samples)
            # Position in [0, 1]
            lhs[:, d] = (perm + offsets) / n_samples
            
        # Map to actual bounds
        x_lhs = lo + lhs * (hi - lo)
        
        # Evaluate all points
        # The problem function is called once per point.
        # We must ensure we don't exceed the budget.
        # We have exactly n_samples points.
        
        F = []
        for i in range(n_samples):
            if time.time() - start_time > 325.0:
                # Safety margin: if we are running out of time, stop and return what we have.
                # However, since we haven't evaluated any yet in the first iteration, 
                # this is unlikely to trigger unless the problem is extremely slow.
                # If it triggers, we return the evaluated points so far.
                break
            F.append(problem(x_lhs[i]))
            
        # The return value is ignored, but we return the points for consistency.
        return x_lhs
# EVOLVE-BLOCK-END
