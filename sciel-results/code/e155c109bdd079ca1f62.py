"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so


class Optimizer:
    """Global grid-based exploration followed by targeted high-resolution local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return float('inf')
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Systematic Global Grid Exploration
        # Use a structured grid to cover the domain [-5, 5]^dim
        # Start with a coarse grid, then refine around promising regions
        
        # Step 1: Very coarse initial sampling (5 levels per dim = 5^dim points)
        # For dim=5: 5^5 = 3125 points, but we only have 500 budget
        # So use 3 levels first: 3^5 = 243 points
        
        # First, evaluate center and corners to get early progress
        # Center
        evaluate(np.zeros(dim))
        
        # Corners of the box [-5, 5]^dim
        if calls_made < budget:
            # Sample a few strategic points: center of each face, midpoints
            # For dim=5, sample points where one coordinate is at boundary, rest at 0
            for i in range(dim):
                for val in [-5, 5]:
                    x = np.zeros(dim)
                    x[i] = val
                    if calls_made < budget:
                        evaluate(x)
        
        # Step 2: Coarse grid with 3 levels per dimension
        # This gives 3^dim points. For dim=5, that's 243 points.
        # We've already evaluated some points, so we'll skip duplicates by tracking
        
        levels_coarse = np.linspace(-5, 5, 3)
        
        # Generate all grid points
        grid_points = np.array(np.meshgrid(*([levels_coarse] * dim))).T.reshape(-1, dim)
        
        # Evaluate remaining grid points (skip if already evaluated approximately)
        evaluated_positions = [np.zeros(dim)]  # center
        for i in range(dim):
            for val in [-5, 5]:
                x = np.zeros(dim)
                x[i] = val
                evaluated_positions.append(x)
        
        for x in grid_points:
            if calls_made >= budget:
                break
            # Skip if this point is already close to an evaluated position
            is_duplicate = False
            for ep in evaluated_positions:
                if np.allclose(x, ep, atol=0.1):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            evaluate(x)
            evaluated_positions.append(x.copy())
        
        # Phase 2: Medium-resolution grid around the best region
        # Take the best point from coarse grid and sample a finer neighborhood
        if calls_made < budget and best_f < float('inf'):
            # Define a local region around best_x
            # Use a smaller box: best_x +/- 2.0 in each dimension
            center = best_x.copy()
            half_width = 2.0
            levels_medium = np.linspace(center - half_width, center + half_width, 5)
            # Clip to domain
            levels_medium = np.clip(levels_medium, -5, 5)
            
            # Generate medium grid
            medium_points = np.array(np.meshgrid(*([levels_medium] * dim))).T.reshape(-1, dim)
            
            # Filter to only points in the domain
            mask = np.all((medium_points >= -5) & (medium_points <= 5), axis=1)
            medium_points = medium_points[mask]
            
            for x in medium_points:
                if calls_made >= budget:
                    break
                # Skip if very close to already evaluated points
                is_duplicate = False
                for ep in evaluated_positions[-20:]:  # Check recent points
                    if np.allclose(x, ep, atol=0.3):
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                evaluate(x)
                evaluated_positions.append(x.copy())
        
        # Phase 3: High-resolution local refinement using Nelder-Mead
        # Start from the best point found
        if calls_made < budget and best_f < float('inf'):
            calls_left = budget - calls_made
            
            def wrapped_func(x):
                nonlocal calls_left
                if calls_left <= 0:
                    return float('inf')
                calls_left -= 1
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            try:
                # Nelder-Mead with tight tolerances for high-resolution search
                res = so.minimize(wrapped_func, best_x, method='Nelder-Mead',
                                  options={
                                      'maxiter': 200,
                                      'xatol': 1e-10,
                                      'fatol': 1e-10,
                                      'initial_simplex': None
                                  })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
