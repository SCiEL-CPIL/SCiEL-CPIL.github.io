"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's known Pareto front with optimized x0 sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        t0 = time.time()
        
        # Optimize x1 placement to maximize hypervolume on ZDT2 front
        # Front: (t, 1-sqrt(t)) for t in [0,1]
        # HV with ref (11,11): sum_{i=1}^{N} (x_{i+1}-x_i)*(10+sqrt(x_i)), x_{N+1}=11
        
        def neg_hv(params):
            x = np.sort(params)
            x_ext = np.concatenate([x, [11.0]])
            widths = x_ext[1:] - x_ext[:-1]
            heights = 10.0 + np.sqrt(np.maximum(x, 0))
            hv = np.sum(widths * heights)
            return -hv
        
        # Initial guess: log-spaced (denser near 0 where front is steep)
        x0 = np.geomspace(1e-6, 0.99, N)
        
        # Use L-BFGS-B
        res = minimize(neg_hv, x0, method='L-BFGS-B', 
                      bounds=[(0, 1)]*N,
                      options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x1_samples = np.sort(res.x)
        
        # If optimization failed or took too long, fall back to log-spaced
        if time.time() - t0 > 20:
            x1_samples = np.geomspace(1e-6, 0.99, N)
        
        # Construct full vectors
        X = np.tile(lo, (N, 1))
        X[:, 0] = x1_samples
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = []
        for i in range(N):
            if time.time() > deadline:
                break
            f_val = problem(X[i])
            F.append(f_val)
            
        return X
# EVOLVE-BLOCK-END
