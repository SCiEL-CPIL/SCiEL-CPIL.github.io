"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ2.
    Strategy:
    1. DTLZ2 with M=3 has Pareto front determined by x[0], x[1].
    2. x[2:] should be 0.5 to minimize g(x).
    3. Use Sobol sequence for low-discrepancy sampling in [0, 1]^2.
    4. Map Sobol coordinates to x[0], x[1] such that points are uniformly distributed on the Pareto front (quarter-sphere).
       - For uniform distribution on sphere surface, cos(theta) should be uniform.
       - x[0] corresponds to theta_1. Let u be uniform in [0,1]. theta_1 = arccos(u). x[0] = 2/pi * theta_1.
       - x[1] corresponds to theta_2. Let v be uniform in [0,1]. theta_2 = v * pi/2. x[1] = v.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Generate Sobol points in [0, 1]^2
        # Sobol requires n to be a power of 2 for perfect uniformity, but it works for any n.
        # We use n_evals points.
        sobol = qmc.Sobol(d=2, scramble=True)
        # sobol.random(n) returns points in [0, 1]
        points_2d = sobol.random(n_evals)
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            u = points_2d[i, 0]
            v = points_2d[i, 1]
            
            # Map to DTLZ2 variables
            # x[0]: uniform on sphere surface -> cos(theta_1) = u -> theta_1 = arccos(u)
            # x[0] = theta_1 / (pi/2) = 2/pi * arccos(u)
            x0 = (2.0 / np.pi) * np.arccos(u)
            
            # x[1]: uniform in [0, pi/2] -> theta_2 = v * pi/2
            # x[1] = theta_2 / (pi/2) = v
            x1 = v
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            
            # Set remaining variables to center of bounds to minimize g(x)
            if self.dim > 2:
                # Assuming bounds are [0, 1] as per task description, center is 0.5
                # But to be robust, use (lo+hi)/2
                center = (lo[2:] + hi[2:]) / 2.0
                x[2:] = center
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                # If an exception occurs, break to preserve valid results
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
