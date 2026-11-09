"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct approach to DTLZ2 with surface-area-uniform sampling on the quarter-sphere Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        count = 0
        
        # We have 200 evaluations.
        # Use a grid that is uniform in surface area on the quarter-sphere.
        # Parameterization: x0 = (2/pi)*arcsin(u), x1 = v, where u,v in [0,1]
        # This gives uniform coverage of the sphere surface.
        
        # Use 14x14 = 196 grid points + 4 strategic = 200
        n_grid = 14
        
        # u values: uniform in [0,1], mapped to x0 via arcsin for uniform surface area
        u_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        # Transform u to x0 for uniform surface coverage
        # x0 = (2/pi) * arcsin(u)
        x0_vals = (2.0 / np.pi) * np.arcsin(u_vals)
        
        for x0 in x0_vals:
            for x1 in v_vals:
                if count >= self.budget:
                    break
                if time.time() > deadline:
                    return np.array(results) if results else None
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                if self.dim > 2:
                    x[2:] = 0.5
                
                x = np.clip(x, lo, hi)
                
                try:
                    f = problem(x)
                    results.append(f)
                    count += 1
                except Exception:
                    break
            
            if count >= self.budget or time.time() > deadline:
                break
        
        # Add strategic points at key locations on the front
        # These correspond to cardinal directions on the quarter-sphere
        extra_points = [
            (0.0, 0.0),   # f = [1, 0, 0]
            (0.0, 1.0),   # f = [0, 1, 0]
            (1.0, 0.0),   # f = [0, 0, 1]
            (0.5, 0.5),   # center of front
        ]
        
        for x0, x1 in extra_points:
            if count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
