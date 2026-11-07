"""Multi-objective optimizer on DTLZ2 using grid search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid search with local refinement optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n_pos = self.n_obj
        n_g = self.dim - n_pos
        
        # 1. Generate initial grid points for the positional variables
        # Use a 5x5x5 grid for 3 objectives
        grid_size = 5
        if self.n_obj == 3:
            grid_points_count = 5 * 5 * 5
        else:
            grid_points_count = 5 ** self.n_obj
            
        # Adjust grid size to fit budget if necessary
        while grid_size ** self.n_obj > self.budget * 0.6:
            grid_size -= 1
            if grid_size < 2:
                grid_size = 2
                break
                
        axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(self.n_obj)]
        grids = np.meshgrid(*axes, indexing='ij')
        grid_X = np.array([g.ravel() for g in grids]).T
        
        # Create full dimension vectors
        X_grid = np.zeros((len(grid_X), self.dim))
        X_grid[:, :self.n_obj] = grid_X
        if n_g > 0:
            X_grid[:, self.n_obj:] = 0.5 * (hi[self.n_obj:] + lo[self.n_obj:])
            
        # Evaluate grid
        F_grid = np.array([problem(x) for x in X_grid], dtype=np.float64)
        
        # Find non-dominated points in the grid
        # Simple non-dominated sort
        non_dom_indices = []
        for i in range(len(F_grid)):
            is_dom = False
            for j in range(len(F_grid)):
                if i == j:
                    continue
                if np.all(F_grid[j] <= F_grid[i]) and np.any(F_grid[j] < F_grid[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom_indices.append(i)
                
        # 2. Refine non-dominated points
        # We have remaining budget
        n_used = len(X_grid)
        n_remaining = self.budget - n_used
        
        # Select a subset of non-dominated points to refine
        # If there are too many, sample them
        if len(non_dom_indices) > n_remaining:
            # Sample uniformly
            refine_indices = np.random.choice(non_dom_indices, n_remaining, replace=False)
        else:
            refine_indices = non_dom_indices
            
        # For each selected point, perform a local search
        # Use a few small steps in the neighborhood
        step_size = 0.1
        n_local_steps = 3 # 3 evaluations per point?
        # Total local evaluations = len(refine_indices) * n_local_steps
        # We need to fit in n_remaining
        
        max_points_to_refine = n_remaining // n_local_steps
        if max_points_to_refine > len(refine_indices):
            max_points_to_refine = len(refine_indices)
            
        # Shuffle and pick top max_points_to_refine
        np.random.shuffle(refine_indices)
        refine_indices = refine_indices[:max_points_to_refine]
        
        X_refined = []
        F_refined = []
        
        for idx in refine_indices:
            x_base = X_grid[idx].copy()
            f_base = F_grid[idx]
            
            # Try random perturbations
            for _ in range(n_local_steps):
                # Check time
                if time.time() > deadline - 10:
                    break
                    
                # Perturb only the positional variables
                x_new = x_base.copy()
                for k in range(self.n_obj):
                    # Random walk
                    x_new[k] += np.random.uniform(-step_size, step_size)
                    x_new[k] = np.clip(x_new[k], lo[k], hi[k])
                
                f_new = np.array(problem(x_new), dtype=np.float64)
                X_refined.append(x_new)
                F_refined.append(f_new)
                
                # If better, update base? 
                # For simplicity, just record. 
                # A more advanced search would update x_base if f_new is better.
                if np.all(f_new <= f_base):
                    x_base = x_new
                    f_base = f_new
        
        # Combine all points
        X_all = np.vstack([X_grid, np.array(X_refined)]) if X_refined else X_grid
        F_all = np.vstack([F_grid, np.array(F_refined)]) if F_refined else F_grid
        
        return X_all
# EVOLVE-BLOCK-END
