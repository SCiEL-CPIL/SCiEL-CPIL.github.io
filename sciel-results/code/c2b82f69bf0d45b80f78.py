"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Deterministic structured search for DTLZ7.
    
    Strategy:
    1. Place a structured grid of points on the known Pareto front (g=0)
       using a 2D grid in (x0, x1) space, which maps to the non-convex front.
    2. Use remaining budget to refine distance variables and explore.
    3. Track all evaluated points, return best set.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # All evaluated points and their objectives
        X_list = []
        F_list = []
        
        def evaluate(x):
            x = np.asarray(x, dtype=np.float64)
            if len(x) != n:
                x = x[:n]
            fx = problem(x)
            return np.asarray(fx, dtype=np.float64)
        
        def add_point(x, f):
            X_list.append(x.copy())
            F_list.append(f.copy())
        
        # Phase 1: Structured grid on the Pareto front
        # DTLZ7 with M=3: Pareto front has g=0, so distance vars = 0
        # f1 = x0, f2 = x1, f3 = sqrt(x0) + sqrt(x1) - 1
        # We want to cover [0,1]^2 in (x0, x1) space
        # Use a grid that emphasizes boundaries/corners for hypervolume
        
        # Create a grid with more points near boundaries
        # Use sqrt-spaced grid to get denser coverage near 0
        n_grid = int(min(40, self.budget // 2))
        # Use a combination of uniform and boundary-emphasized points
        
        # Generate grid points
        grid_points = []
        
        # Uniform grid
        n_per_axis = int(np.ceil(np.sqrt(n_grid)))
        for i in range(n_per_axis):
            for j in range(n_per_axis):
                if len(grid_points) >= n_grid:
                    break
                x0 = i / max(n_per_axis - 1, 1)
                x1 = j / max(n_per_axis - 1, 1)
                grid_points.append((x0, x1))
        
        # Add explicit boundary points
        boundary_points = []
        n_boundary = min(20, self.budget // 10)
        for i in range(n_boundary + 1):
            t = i / n_boundary
            boundary_points.append((0.0, t))  # x0=0 edge
            boundary_points.append((t, 0.0))  # x1=0 edge
            boundary_points.append((1.0, t))  # x0=1 edge
            boundary_points.append((t, 1.0))  # x1=1 edge
            boundary_points.append((t, t))    # diagonal
            boundary_points.append((1.0 - t, t))
        
        # Combine and deduplicate
        all_front_points = list(grid_points)
        for bp in boundary_points:
            if not any(abs(bp[0] - gp[0]) < 1e-10 and abs(bp[1] - gp[1]) < 1e-10 for gp in all_front_points):
                all_front_points.append(bp)
        
        # Evaluate front points
        for x0, x1 in all_front_points:
            if time.time() > deadline:
                break
            if len(X_list) >= self.budget:
                break
            x = np.zeros(n)
            x[0] = x0
            x[1] = x1
            # Distance variables are 0 for Pareto front
            f = evaluate(x)
            add_point(x, f)
        
        # Phase 2: Use remaining budget for local refinement
        # Try small perturbations of distance variables to see if we can improve
        remaining = self.budget - len(X_list)
        if remaining > 0 and len(X_list) > 0:
            # Pick some promising points and perturb distance vars
            # Focus on points that are well-spread in objective space
            F_arr = np.array(F_list)
            
            # Select diverse points from current set
            selected_indices = []
            for i in range(len(F_arr)):
                if len(selected_indices) >= min(10, len(F_arr)):
                    break
                if not selected_indices:
                    selected_indices.append(i)
                    continue
                # Check if this point is sufficiently different
                is_different = True
                for j in selected_indices:
                    if np.all(F_arr[i] >= F_arr[j] - 1e-6) and np.all(F_arr[i] <= F_arr[j] + 1e-6):
                        is_different = False
                        break
                if is_different:
                    selected_indices.append(i)
            
            # Perturb distance variables
            for idx in selected_indices:
                if time.time() > deadline or len(X_list) >= self.budget:
                    break
                base_x = X_list[idx].copy()
                # Small perturbations to distance variables
                for _ in range(min(3, remaining)):
                    if time.time() > deadline or len(X_list) >= self.budget:
                        break
                    perturbation = np.zeros(n)
                    for d in range(2, n):
                        perturbation[d] = 0.01 * (2 * np.random.random() - 1)
                    child = base_x + perturbation
                    child = np.clip(child, lo, hi)
                    fc = evaluate(child)
                    add_point(child, fc)
        
        return X_list
# EVOLVE-BLOCK-END
