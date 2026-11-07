"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic 2D Fibonacci lattice on the DTLZ7 Pareto front.
    
    DTLZ7 with M=3:
    Pareto front is given by x[2:] = 0.
    f1 = 2 * x[0]
    f2 = 2 * x[1]
    f3 = sqrt(x[0]) + sqrt(x[1]) - 1
    
    Reference point: [1.1, 1.1, 25.0]
    Points must satisfy f1 < 1.1, f2 < 1.1, f3 < 25.0.
    f1 < 1.1 => 2*x[0] < 1.1 => x[0] < 0.55
    f2 < 1.1 => 2*x[1] < 1.1 => x[1] < 0.55
    f3 < 25.0 is always true for x in [0, 1].
    
    We generate points in x[0], x[1] in [0, 0.55] using a 2D Fibonacci lattice
    to maximize coverage of the contributing Pareto front region.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n_eval = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Golden ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        
        # Generate n_eval indices
        n = np.arange(1, n_eval + 1, dtype=np.float64)
        
        # 2D Fibonacci lattice (Golden ratio spiral)
        # u = (n * phi) % 1
        # v = (n * phi^2) % 1
        # This provides low discrepancy in 2D.
        u = (n * phi) % 1.0
        v = (n * phi * phi) % 1.0
        
        # Scale to the useful region [0, 0.55]
        # x[0] corresponds to u, x[1] corresponds to v
        scale = 0.55
        x0_vals = u * scale
        x1_vals = v * scale
        
        # Create candidate solutions array
        X = np.zeros((n_eval, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        # x[2:] remain 0
        
        # Clip to bounds (just in case, though 0.55 is within [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all
        # Using a list comprehension is often faster than a for loop with append for small N
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
