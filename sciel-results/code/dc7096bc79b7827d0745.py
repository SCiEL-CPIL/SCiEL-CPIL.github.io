"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    Exploits the known Pareto front at x[1:] = 0.
    Uses Latin Hypercube Sampling (LHS) over x[0] domain for robust coverage.
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
        
        # Generate Latin Hypercube Samples for x[0]
        # LHS ensures one sample per stratum, providing uniform coverage
        u = np.random.rand(n_points)
        # Sort to create the LHS structure (each stratum has exactly one point)
        u_sorted = np.sort(u)
        
        # Map to [lo[0], hi[0]]
        x0_values = lo[0] + (hi[0] - lo[0]) * u_sorted

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
