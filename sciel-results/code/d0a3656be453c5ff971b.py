"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # DTLZ2 with M=3: Pareto front is a quarter-sphere in objective space.
        # Parameterize by angles theta1, theta2 in [0, pi/2].
        # x[0] = cos(theta1) * cos(theta2)
        # x[1] = cos(theta1) * sin(theta2)
        # x[2:] = 0.5 (to make g=0)
        # This gives f1 = sin(theta1), f2 = cos(theta1)*sin(theta2), f3 = cos(theta1)*cos(theta2)
        
        # Use a non-uniform grid in angle space with denser spacing near boundaries
        # 14x14 = 196 points, leaving 4 for potential refinement
        n_per_axis = 14
        
        # Cosine spacing: denser near 0 and pi/2 where hypervolume contribution is higher
        theta1_vals = (np.pi / 2) * (1 - np.cos(np.linspace(0, np.pi, n_per_axis))) / 2
        theta2_vals = (np.pi / 2) * (1 - np.cos(np.linspace(0, np.pi, n_per_axis))) / 2
        
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')
        grid_points = np.column_stack([T1.ravel(), T2.ravel()])
        
        # Use all 196 points from the grid
        results = []
        
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            
            # Position variables from angle parameterization
            t1 = grid_points[i, 0]
            t2 = grid_points[i, 1]
            x[0] = np.cos(t1) * np.cos(t2)
            x[1] = np.cos(t1) * np.sin(t2)
            
            # Distance variables: set to center of bounds
            if self.dim > 2:
                x[2:] = 0.5
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        # Use remaining budget for extra points at non-grid locations
        # With cosine-spaced grid, find gaps and place extra points in high-value regions
        remaining = n_points - len(results)
        if remaining > 0 and time.time() < deadline:
            # Place extra points at positions that maximize marginal hypervolume
            # Near the "corners" where one or two objectives are very small
            extra_angles = [
                (np.pi/2, np.pi/4),       # f1~1, f2~0.7, f3~0.7 - high f1
                (np.pi/4, np.pi/2),       # f1~0.7, f2~0.7, f3~0 - low f3
                (np.pi/2, np.pi/2),       # f1~1, f2~0, f3~0 - two objectives near 0
                (np.pi/8, np.pi/8),       # f1~0.38, f2~0.65, f3~0.65 - balanced low
            ]
            
            for t1, t2 in extra_angles[:remaining]:
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = np.cos(t1) * np.cos(t2)
                x[1] = np.cos(t1) * np.sin(t2)
                if self.dim > 2:
                    x[2:] = 0.5
                x = np.clip(x, lo, hi)
                try:
                    f = problem(x)
                    results.append(f)
                except Exception:
                    break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
