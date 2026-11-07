"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


class Optimizer:
    """Nested Grid Zoom: Global coarse grid followed by iterative local grid refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Phase 1: Global Coarse Grid
        # 3 levels per dimension to cover [-5, 5]
        # Points: -5, 0, 5
        levels_global = np.linspace(-5, 5, 3)
        grid_points = np.array(np.meshgrid(*([levels_global] * self.dim))).T.reshape(-1, self.dim)
        
        # Sort grid points by distance to the origin (0,0,0,0,0)
        # This ensures we evaluate the center and nearby points first,
        # improving early AOCC if the minimum is near the origin.
        center = np.zeros(self.dim)
        distances = np.linalg.norm(grid_points - center, axis=1)
        sorted_indices = np.argsort(distances)
        grid_points = grid_points[sorted_indices]
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate global grid
        for x in grid_points:
            if calls_made >= self.budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return None, float('inf')

        # Phase 2: Local Grid Refinement (Zoom)
        # We will perform iterative local grid searches with decreasing step sizes.
        # Step size starts at the global grid step (5.0) and divides by 3 each time.
        # Number of levels per dimension is fixed at 3 for the local grids.
        
        current_x = best_x
        current_f = best_f
        step_size = 5.0  # Global step: (5 - (-5)) / (3 - 1) = 5.0
        
        while calls_made < self.budget:
            # Generate local grid around current_x
            # Levels: current_x - step, current_x, current_x + step
            # Clip to bounds [-5, 5]
            local_levels = []
            for i in range(self.dim):
                local_levels.append(np.array([
                    np.clip(current_x[i] - step_size, -5, 5),
                    current_x[i],
                    np.clip(current_x[i] + step_size, -5, 5)
                ]))
            
            local_grid_points = np.array(np.meshgrid(*local_levels)).T.reshape(-1, self.dim)
            
            # Sort local grid points by distance to current_x
            distances_local = np.linalg.norm(local_grid_points - current_x, axis=1)
            sorted_indices_local = np.argsort(distances_local)
            local_grid_points = local_grid_points[sorted_indices_local]

            grid_size = local_grid_points.shape[0]
            
            if calls_made + grid_size <= self.budget:
                # Evaluate full local grid
                for x in local_grid_points:
                    fx = func(x)
                    calls_made += 1
                    if fx < current_f:
                        current_f = fx
                        current_x = x.copy()
            else:
                # If we don't have enough for the full grid, evaluate as many as possible
                for x in local_grid_points:
                    if calls_made >= self.budget:
                        break
                    fx = func(x)
                    calls_made += 1
                    if fx < current_f:
                        current_f = fx
                        current_x = x.copy()
                break # No more budget for further zooms

            # Reduce step size for next iteration
            step_size /= 3.0
            
            # If step size becomes too small, break
            if step_size < 1e-6:
                break

        return current_x, current_f
# EVOLVE-BLOCK-END
