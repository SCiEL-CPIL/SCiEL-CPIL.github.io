import numpy as np
import time
from itertools import product

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Grid search on the first 3 variables (shape parameters for WFG9)
        # For WFG9, the first 3 variables determine the Pareto front shape.
        # Variables 4-12 affect convergence but not the front shape.
        # We'll use a 6x6x6 grid = 216 points for the first 3 variables.
        # This gives better coverage of the Pareto front.
        grid_size = 6
        N_grid = grid_size ** 3  # 216 points
        
        if self.budget < N_grid + 10:
            # If budget is too small, reduce grid size
            grid_size = 5
            N_grid = grid_size ** 3  # 125 points
            if self.budget < N_grid + 10:
                grid_size = 4
                N_grid = grid_size ** 3  # 64 points
        
        # Generate grid points for first 3 variables
        # Use linspace to cover the range evenly
        grid_vals = np.linspace(lo[0], hi[0], grid_size)
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = mid.copy()
                    x[0] = grid_vals[i]
                    x[1] = grid_vals[j]
                    x[2] = grid_vals[k]
                    # Keep variables 3-11 at midpoint (convergence variables)
                    grid_points.append(x)
        
        grid_points = np.array(grid_points)
        
        # Evaluate all grid points
        F_grid = np.array([problem(x) for x in grid_points])
        used = N_grid
        
        # Maintain archive
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            # Check if f is dominated by any in archive
            dominated = False
            for _, f_arch in zip(archive_X, archive_F):
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    dominated = True
                    break
            if dominated:
                return
            
            # Remove points in archive dominated by f
            new_archive_X = []
            new_archive_F = []
            for x_arch, f_arch in zip(archive_X, archive_F):
                if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                    new_archive_X.append(x_arch)
                    new_archive_F.append(f_arch)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
        
        # Initialize archive with grid points
        for i in range(N_grid):
            update_archive(grid_points[i], F_grid[i])
        
        # Phase 2: Local search on non-dominated points
        # Focus on improving convergence by optimizing variables 3-11
        N_search = self.budget - used
        
        if N_search <= 0:
            return archive_X
        
        # For each non-dominated point, do local search on convergence variables
        # Use biased steps toward the midpoint or toward better convergence
        
        for i in range(N_search):
            if time.time() > deadline:
                break
            
            if len(archive_X) == 0:
                break
            
            # Pick a random non-dominated point
            parent_idx = np.random.randint(len(archive_X))
            x_parent = np.array(archive_X[parent_idx])
            
            # Local search: perturb only convergence variables (indices 3-11)
            # with smaller steps to improve convergence
            x = x_parent.copy()
            
            # Adaptive step size: larger early, smaller later
            progress = i / max(1, N_search - 1)
            step_size = 0.1 * (1 - progress) + 0.01
            
            # Perturb convergence variables
            for d in range(3, dim):
                if np.random.random() < 0.5:  # Only perturb some variables
                    x[d] += np.random.normal(0, 1) * (hi[d] - lo[d]) * step_size
                    x[d] = np.clip(x[d], lo[d], hi[d])
            
            f = problem(x)
            used += 1
            
            update_archive(x, f)
        
        return archive_X
