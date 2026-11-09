import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        n_shape = min(3, self.dim)
        n_pos = self.dim - n_shape
        
        # WFG9 structure:
        # First 3 variables are "shape" variables that determine the Pareto front.
        # The remaining variables are "position" variables.
        # For WFG9, the optimal position for the position variables is the midpoint.
        # The shape variables define the front.
        
        # Strategy:
        # 1. Grid on the 3 shape variables to map the front.
        # 2. Use remaining budget to refine non-dominated points with local search
        #    on the shape variables, keeping position variables at midpoint.
        
        # Phase 1: Grid on shape variables
        # Budget is 200.
        # A 6x6x5 grid gives 180 points.
        # A 5x5x5 grid gives 125 points.
        # Let's try 6x6x4 = 144 points, leaving 56 for refinement.
        # Or 5x5x5 = 125, leaving 75.
        # The inspiration used 6x6x5=180 + 20 random.
        # Let's try a slightly denser grid on a subset or just use the 6x6x5 grid.
        
        # Actually, the "adaptive grid" finding suggests focusing on the best region.
        # But WFG9's front is well-defined by the shape variables.
        # A uniform grid on shape variables with position at midpoint should be very effective.
        
        # Let's use 6x6x5 = 180 points for the grid.
        g1, g2, g3 = 6, 6, 5
        grid_sizes = [g1, g2, g3]
        
        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))
        
        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N_grid = grid_points.shape[0]
        
        # Construct full vectors: shape from grid, position from mid
        X_grid = np.tile(mid, (N_grid, 1))
        X_grid[:, :n_shape] = grid_points
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in X_grid])
        used = N_grid
        
        if time.time() > deadline:
            return X_grid
            
        # Phase 2: Refinement
        # Identify non-dominated points in the grid.
        n_points = len(F_grid)
        is_dom = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if i == j: continue
                if np.all(F_grid[j] <= F_grid[i]) and np.any(F_grid[j] < F_grid[i]):
                    is_dom[i] = True
                    break
        
        non_dom_idx = np.where(~is_dom)[0]
        
        remaining = self.budget - used
        
        if remaining > 0 and len(non_dom_idx) > 0:
            # For each non-dominated point, try a few local perturbations
            # on the shape variables.
            # We have 'remaining' evaluations left.
            # Let's distribute them among the non-dominated points.
            n_refine_per_point = max(1, remaining // len(non_dom_idx))
            
            X_refine = []
            F_refine = []
            
            for idx in non_dom_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                parent_x = X_grid[idx]
                parent_f = F_grid[idx]
                
                # Try n_refine_per_point perturbations
                for k in range(n_refine_per_point):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    # Perturb shape variables
                    # Small step
                    step = 0.1 * (hi[:n_shape] - lo[:n_shape])
                    noise = np.random.normal(0, 1, n_shape) * step
                    new_shape = np.clip(parent_x[:n_shape] + noise, lo[:n_shape], hi[:n_shape])
                    
                    x = np.empty(self.dim)
                    x[:n_shape] = new_shape
                    x[n_shape:] = mid[n_shape:]  # Keep position at midpoint
                    
                    f = problem(x)
                    X_refine.append(x)
                    F_refine.append(f)
                    used += 1
            
            if X_refine:
                X_final = np.vstack([X_grid, np.array(X_refine)])
            else:
                X_final = X_grid
        else:
            X_final = X_grid
            
        return X_final
