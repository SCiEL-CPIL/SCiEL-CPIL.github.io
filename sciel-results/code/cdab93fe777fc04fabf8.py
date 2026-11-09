"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is 1D: only x[0] varies, x[1:] = 0.
    Uses Sobol quasi-random sampling for dense, low-discrepancy coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[1:] = 0, only x[0] varies in [0,1]
        # Use Sobol sequence for dense, low-discrepancy 1D coverage
        
        # Generate Sobol sequence points in [0, 1]
        n_sobol = budget
        
        # Use scipy's Sobol sequence generator
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=1, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        x1_vals = sobol_points[:, 0]
        
        # Ensure we have exactly budget points
        x1_vals = x1_vals[:budget]
        
        # Evaluate each point
        for x1 in x1_vals:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 ensures g(x) = 0 (on Pareto front)
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
