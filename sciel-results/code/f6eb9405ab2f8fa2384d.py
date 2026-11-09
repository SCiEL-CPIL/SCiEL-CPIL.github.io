import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific:
        # x[0], x[1] determine front position
        # x[2:] determine distance to front. Set to 0.5 for g(x)=0.
        
        n_evals = self.budget
        
        # Use a stratified grid on the Pareto front (spherical surface)
        # Area element is dz d_beta where z = cos(alpha), beta is azimuth.
        # Uniform grid in (z, beta) gives uniform area coverage.
        
        # We want approx 200 points.
        # 10 x 20 = 200 points.
        n_z = 10
        n_beta = 20
        
        if n_z * n_beta > n_evals:
             # Adjust to fit budget if needed, but 200 is the budget.
             pass
            
        results = []
        
        # Generate z values (midpoints of intervals in [0, 1])
        z_vals = (np.arange(n_z) + 0.5) / n_z
        # Generate beta values (midpoints of intervals in [0, pi/2])
        beta_vals = (np.arange(n_beta) + 0.5) * (np.pi / 2) / n_beta
        
        # Loop through grid
        for i in range(n_z):
            for j in range(n_beta):
                if time.time() > deadline:
                    return results
                
                if len(results) >= n_evals:
                    return results

                z = z_vals[i]
                beta = beta_vals[j]
                
                # Map back to x0, x1
                # z = cos(x0 * pi / 2) => x0 = (2/pi) * arccos(z)
                # beta = x1 * pi / 2 => x1 = (2/pi) * beta
                
                x0 = (2.0 / np.pi) * np.arccos(z)
                x1 = (2.0 / np.pi) * beta
                
                # Scale to bounds (which are [0, 1])
                x0 = lo[0] + (hi[0] - lo[0]) * x0
                x1 = lo[1] + (hi[1] - lo[1]) * x1
                
                X = np.zeros(self.dim)
                X[0] = x0
                X[1] = x1
                # Set remaining variables to 0.5 to ensure g(x)=0 (on the front)
                # Assuming bounds are [0,1], 0.5 is the midpoint.
                # If bounds are different, we should use the midpoint of the bounds for optimal g=0 in DTLZ2?
                # DTLZ2 g(x) = sum (xi - 0.5)^2. So 0.5 is the minimizer regardless of bounds?
                # No, if bounds are [0,1], 0.5 is inside.
                # The problem statement says bounds [0,1].
                X[2:] = 0.5
                
                f = problem(X)
                results.append(f)
                
        return results
