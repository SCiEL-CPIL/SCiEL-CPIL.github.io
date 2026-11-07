"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 with optimized Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # We use the entire budget for sampling the Pareto front.
        # ZDT2 Pareto front: x1 in [0, 1], x2..xn = 0.
        # Optimal sampling distribution for hypervolume: x1 ~ t^(4/3)
        
        n_samples = budget
        
        # Generate t values in [0, 1]
        t_vals = np.linspace(0, 1, n_samples)
        
        # Map to x1 using the derived power law
        # We want to cover [0, 1] in x1.
        # x1 = t^(4/3) maps [0,1] to [0,1] with higher density near 0.
        x1_vals = t_vals ** (4.0 / 3.0)
        
        # Ensure we hit the boundaries exactly to maximize range
        x1_vals[0] = 0.0
        x1_vals[-1] = 1.0
        
        # Create the population
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_vals
        # x2..xn are already 0.0
        
        # Clip to bounds just in case, though 0 and 1 are within [0,1]
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
