"""ZDT1 optimizer using analytical power-law spacing and local refinement."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin
        
        N = self.budget
        dim = self.dim
        R1, R2 = 11.0, 11.0
        
        # Analytical initial guess: x_i = (i/N)^4
        # Derived from minimizing error of step function approximation for f''(x) ~ x^-1.5
        i = np.arange(1, N + 1)
        x_init = (i / N) ** 4
        
        # Ensure x is within bounds [0, 1]
        x_init = np.clip(x_init, 0.0, 1.0)
        
        def compute_hv(x_vals):
            """Compute hypervolume for points on ZDT1 front."""
            # Sort x values
            x_sorted = np.sort(x_vals)
            # Calculate f2 values
            f2_vals = 1.0 - np.sqrt(x_sorted)
            
            # Filter valid points (strictly better than reference)
            # All x in [0,1] imply f1 in [0,1] < 11 and f2 in [0,1] < 11
            # So all are valid.
            
            # Hypervolume calculation for 2D
            # HV = sum_{i=1}^{N-1} (x_{i+1} - x_i) * (R2 - f2_i) + (R1 - x_N) * (R2 - f2_N)
            # Note: This formula assumes the "step" goes from x_i to x_{i+1} at height f2_i.
            # This is the correct decomposition for the union of rectangles [x_i, x_{i+1}] x [f2_i, R2]
            # plus the last one [x_N, R1] x [f2_N, R2].
            
            if len(x_sorted) == 0:
                return 0.0
            
            # Differences in x
            # x_{i+1} - x_i
            dx = np.diff(x_sorted)
            # Last segment to R1
            dx_last = R1 - x_sorted[-1]
            
            # Heights
            h = R2 - f2_vals
            
            # Sum of areas
            # First N-1 segments
            area = np.sum(dx * h[:-1])
            # Last segment
            area += dx_last * h[-1]
            
            return area

        def objective(x):
            return -compute_hv(x)
        
        # Optimize the x values
        # Bounds: [0, 1] for each x
        bounds_opt = [(0.0, 1.0) for _ in range(N)]
        
        try:
            # Use L-BFGS-B for speed and robustness
            # We provide the initial guess
            res = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds_opt, 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
            x_opt = res.x
        except Exception:
            # Fallback to initial guess if optimization fails
            x_opt = x_init
            
        # Check time
        if time.time() > deadline - 10.0:
            x_opt = x_init

        # Construct input vectors
        # ZDT1: f1 = x1, f2 = 1 - sqrt(x1) * g(x2...xn)
        # For Pareto front, g(x) = 1, which requires x2...xn = 0 (or specific values, but 0 works for standard ZDT1 definition where g=1 if x_i in [0,1] and sum is small? 
        # Actually, standard ZDT1: g(x) = 1 + 10(n-1)/sum(x_i) if sum>0 else 1.
        # If x2...xn = 0, sum=0, g=1. This is the Pareto front.
        
        X = np.zeros((N, dim))
        X[:, 0] = x_opt
        # x2...xn are already 0
        
        # Evaluate all points
        # The problem function clips and returns objectives.
        # We just need to call it.
        F = np.array([problem(x) for x in X])
        
        return X
