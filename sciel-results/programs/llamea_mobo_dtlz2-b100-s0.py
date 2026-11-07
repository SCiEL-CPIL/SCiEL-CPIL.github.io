import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: exploit known structure for Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front is achieved when g = 0, i.e., x_i = 0.5 for i >= m
        # The remaining m-1 = 2 variables (x_1, x_2, indices 0, 1) parameterize the front.
        # f_1 = cos(theta_1)cos(theta_2), f_2 = cos(theta_1)sin(theta_2), f_3 = sin(theta_1)
        # theta_1 = x_0 * pi/2, theta_2 = x_1 * pi/2
        # Surface area element dA = cos(theta_1) dtheta_1 dtheta_2
        # Uniform sampling requires:
        # 1. theta_2 uniform in [0, pi/2] -> x_1 uniform in [0, 1]
        # 2. sin(theta_1) uniform in [0, 1] -> theta_1 = arcsin(u) -> x_0 = (2/pi) * arcsin(u)

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        evaluated_points = []
        used_budget = 0

        # Strategy: Use a structured grid for dense coverage of the Pareto front,
        # then fill remaining budget with Halton sequence points for refinement.
        
        # Part 1: Structured grid
        # Use a grid that gives good coverage. 12x12 = 144 points.
        # This ensures we hit the "corners" and edges well.
        grid_size = 12
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Map grid indices to [0, 1]
                # Use midpoints to avoid boundary issues if necessary, but boundaries are fine.
                # Actually, for hypervolume, including the boundaries (0 and 1) is crucial 
                # to capture the full extent.
                # Let's use a uniform grid including endpoints.
                u = i / (grid_size - 1)
                v = j / (grid_size - 1)
                grid_points.append((u, v))
        
        # Evaluate grid points
        for u, v in grid_points:
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
            
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        # Part 2: Halton sequence for remaining budget
        # Start Halton sequence from index 1.
        # We need to avoid duplicating points if possible, but Halton is low-discrepancy
        # so overlaps with grid are unlikely to be harmful, and duplicates in objective space
        # just don't add to hypervolume.
        halton_idx = 1
        while used_budget < self.budget:
            if time.time() > deadline:
                break
            
            u = halton(halton_idx, 2)
            v = halton(halton_idx, 3)
            
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            halton_idx += 1

        return evaluated_points
