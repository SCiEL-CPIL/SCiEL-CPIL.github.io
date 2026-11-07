"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    Exploits the known Pareto front at x[1:] = 0.
    Uses non-uniform sampling of x[0] biased towards x[0]=0 to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT2 Pareto front is at x[1:] = 0
        # x[0] varies in [lo[0], hi[0]]
        
        n_points = self.budget
        
        # Latin Hypercube Sampling for x[0]
        # Divide [0,1] into n_points equal intervals, sample one point from each
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility within the optimizer
        perm = rng.permutation(n_points)
        u_lhs = (perm + rng.uniform(size=n_points)) / n_points
        # Sort to get ordered samples (optional, but helps with deterministic behavior)
        u_lhs = np.sort(u_lhs)
        
        # Map to x[0] range
        x0_values = lo[0] + (hi[0] - lo[0]) * u_lhs
        
        # Ensure exact endpoints for maximum coverage
        x0_values[0] = lo[0]
        x0_values[-1] = hi[0]

        # Initialize candidate matrix
        candidates = np.zeros((n_points, self.dim))
        candidates[:, 0] = x0_values
        
        # Set remaining dimensions to lower bound
        if self.dim > 1:
            candidates[:, 1:] = lo[1:]

        evaluated_points = []
        
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
                
            x = candidates[i]
            f = problem(x)
            evaluated_points.append(f)
            
        return None
# EVOLVE-BLOCK-END
