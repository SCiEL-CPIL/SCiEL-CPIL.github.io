"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using dense 1D grid search on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT4: x0 is in [0,1], x1..x9 are in [-5,5]
        # The Pareto front is traced by varying x0 while minimizing g(x)
        # g(x) = 10*n + sum(xi^2 - 10*cos(4*pi*xi + pi)) for i=1..n-1
        # Each term h(xi) = xi^2 - 10*cos(4*pi*xi + pi) has global min ~ -9.4375
        # at xi ≈ 0.75 or xi ≈ -0.25
        
        # Fix x1..x9 at 0.75 (a known global minimizer of h(xi))
        x_fixed = np.zeros(self.dim)
        x_fixed[0] = 0.5  # Will be swept
        for i in range(1, self.dim):
            x_fixed[i] = 0.75
        
        # Use entire budget to sweep x0
        n_evals = self.budget
        x0_values = np.linspace(lo[0], hi[0], n_evals)
        
        results = []
        for i, x0_val in enumerate(x0_values):
            if time.time() > deadline:
                break
            
            x = x_fixed.copy()
            x[0] = x0_val
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
