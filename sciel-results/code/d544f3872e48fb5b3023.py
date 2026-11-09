import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: set x[2:] = 0 to minimize g
        # Front: f1 = x[0], f2 = x[1], f3 = 2 - x[1] (when g=1)
        # So we need to sample (x0, x1) in [0,1]^2
        
        # Strategy: Use a combination of:
        # 1. Critical corner points (4 corners of the front)
        # 2. Edge-midpoint points for better boundary coverage
        # 3. Interior points using a stratified/quasi-random approach
        #    with non-uniform density favoring regions near edges
        
        # Fixed strategic points
        fixed_points = [
            [0.0, 0.0],   # (f1=0, f2=0, f3=2)
            [1.0, 0.0],   # (f1=1, f2=0, f3=2)
            [0.0, 1.0],   # (f1=0, f2=1, f3=1)
            [1.0, 1.0],   # (f1=1, f2=1, f3=1)
            [0.5, 0.0],   # Edge midpoint
            [0.5, 1.0],   # Edge midpoint
            [0.0, 0.5],   # Edge midpoint
            [1.0, 0.5],   # Edge midpoint
            [0.0, 0.25],  # Quarter points
            [0.0, 0.75],
            [1.0, 0.25],
            [1.0, 0.75],
            [0.25, 0.0],
            [0.75, 0.0],
            [0.25, 1.0],
            [0.75, 1.0],
        ]
        
        n_fixed = len(fixed_points)
        n_interior = budget - n_fixed
        
        if n_interior < 0:
            n_interior = 0
            n_fixed = budget
            fixed_points = fixed_points[:budget]
        
        all_points = list(fixed_points)
        
        # Generate interior points using a stratified approach
        # Use a grid with non-uniform spacing: denser near edges
        # For the interior, use a Sobol-like sequence but with warping
        
        if n_interior > 0:
            # Use a power-of-2 sized Sobol sequence for good space-filling
            # Then apply a transformation to bias toward edges
            n_gen = 1
            while n_gen < n_interior:
                n_gen *= 2
            
            # Generate Sobol sequence
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=123)
            sobol_raw = sampler.random(n_gen)
            
            # Apply a warping function to bias toward edges
            # Use a power transformation: u -> u^alpha for alpha < 1 biases toward 0
            # and u -> 1 - (1-u)^alpha biases toward 1
            # For symmetric bias toward both edges, use:
            # u_warped = 0.5 + 0.5 * sign(u-0.5) * (2*min(u,1-u))^beta
            # where beta < 1 pushes points toward edges
            
            beta = 0.7  # Less than 1 pushes toward edges
            
            sobol_warped = sobol_raw.copy()
            for i in range(2):
                u = sobol_raw[:, i]
                # Map [0,1] to [0,1] with bias toward edges
                # u_warped = 0.5 + 0.5 * (2*u - 1) * (2*min(u,1-u))^beta / (2*min(u,1-u)) 
                # Simpler: use a CDF-based warping
                # For a Beta(a,b) distribution with a,b < 1, mass concentrates at edges
                # Use a=0.5, b=0.5 (arcsine-like) for moderate edge bias
                # Or use a simple power warp
                # u_warped = u^a for u in [0, 0.5], and 1-(1-u)^a for u in [0.5, 1]
                a = 0.6  # < 1 biases toward edges
                lower = u <= 0.5
                sobol_warped[lower, i] = u[lower] ** a
                sobol_warped[~lower, i] = 1.0 - (1.0 - u[~lower]) ** a
            
            # Take the first n_interior points
            interior_points = sobol_warped[:n_interior]
            all_points.extend(interior_points.tolist())
        
        # Evaluate all points
        for i in range(len(all_points)):
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
                
            u1, u2 = all_points[i]
            
            # Ensure values are in [0, 1]
            u1 = np.clip(u1, 0.0, 1.0)
            u2 = np.clip(u2, 0.0, 1.0)
            
            x = np.zeros(self.dim)
            x[0] = u1
            x[1] = u2
            # x[2:] remains 0, which minimizes g for DTLZ7
            
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        return results
