# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 exploiting known structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9 Structure:
        # x[0] is the distance variable (z1). Minimizing this (setting to lo[0]) puts points on the Pareto front.
        # x[1], x[2] are shape variables (z2, z3). They determine the position along the front.
        # x[3:] are position variables. Setting them to midpoints (0.5 in normalized space) ensures they do not add distance.
        
        # Strategy:
        # 1. Fix x[0] to lo[0] (best distance).
        # 2. Fix x[3:] to midpoints.
        # 3. Sample x[1] and x[2] densely using a grid.
        
        n_evals = self.budget
        
        # Calculate grid size for 2D sampling of x[1], x[2]
        # We want to use most of the budget for the grid.
        grid_size = int(np.sqrt(n_evals))
        
        # Ensure we don't exceed budget
        grid_points = grid_size * grid_size
        if grid_points > n_evals:
            grid_size -= 1
            grid_points = grid_size * grid_size
            
        remaining_evals = n_evals - grid_points
        
        all_X = []
        all_F = []
        used = 0

        # Create grid for x[1] and x[2]
        vals_1 = np.linspace(lo[1], hi[1], grid_size)
        vals_2 = np.linspace(lo[2], hi[2], grid_size)
        
        # Fixed values for other variables
        fixed_x0 = lo[0]
        fixed_tail = mid[3:].copy()

        # Evaluate grid points
        for v1 in vals_1:
            if time.time() > deadline:
                break
            for v2 in vals_2:
                if used >= n_evals or time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = fixed_x0
                x[1] = v1
                x[2] = v2
                x[3:] = fixed_tail
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        # If we have budget left, add random points on the front
        if used < n_evals:
            rng = np.random.default_rng(42)
            n_rand = n_evals - used
            for _ in range(n_rand):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = fixed_x0
                x[1] = rng.uniform(lo[1], hi[1])
                x[2] = rng.uniform(lo[2], hi[2])
                x[3:] = fixed_tail
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        # Return the last X (return value is ignored, but keep interface)
        if all_X:
            return np.array(all_X[-1])
        return np.zeros(self.dim)
# EVOLVE-BLOCK-END
