import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            return func(np.clip(x, -5, 5))
        
        # Phase 1: Hierarchical Grid Search
        # Exploit the periodic structure of Katsuura function
        # Start with a coarse grid, then refine the best region
        
        # Level 1: Coarse grid 3^5 = 243 points
        # This gives us a broad overview of the landscape
        grid_sizes = [3, 5, 7]
        
        for level, gsize in enumerate(grid_sizes):
            if used_calls >= max_calls or time.time() > deadline:
                break
            
            # Determine the region to search
            if level == 0:
                # First level: full domain
                lower = np.array([-5.0] * self.dim)
                upper = np.array([5.0] * self.dim)
            else:
                # Subsequent levels: refine around the best point found
                # Use a fraction of the previous grid cell size
                # The previous grid cell size was (upper - lower) / gsize
                # We want to focus on the best cell, so we use a smaller region
                # Actually, let's just use the best point as center and shrink the region
                # But we need to be careful about boundaries
                # Let's use a fixed fraction of the domain for refinement
                # For level 1 (gsize=5), use a region of size 2.0 centered on best
                # For level 2 (gsize=7), use a region of size 1.0 centered on best
                shrink_factor = 1.0 / (gsize - 1)  # Rough approximation
                half_width = 5.0 * shrink_factor
                center = best_x
                lower = np.maximum(center - half_width, -5.0)
                upper = np.minimum(center + half_width, 5.0)
                # Ensure lower < upper
                if np.any(lower >= upper):
                    break
            
            # Generate grid points
            # For each dimension, create gsize points
            grid_points = []
            for idx in np.ndindex(*([gsize] * self.dim)):
                x = np.zeros(self.dim)
                for d in range(self.dim):
                    x[d] = lower[d] + (upper[d] - lower[d]) * idx[d] / (gsize - 1)
                grid_points.append(x)
            
            # Evaluate all grid points
            for p in grid_points:
                if used_calls >= max_calls or time.time() > deadline:
                    break
                try:
                    fp = safe_func(p)
                    if fp < best_f:
                        best_f = fp
                        best_x = p.copy()
                except RuntimeError:
                    break
            
            # If we ran out of budget or time, stop
            if used_calls >= max_calls or time.time() > deadline:
                break
        
        # Phase 2: Local Optimization from the best point found by grid search
        if best_x is not None and used_calls < max_calls and time.time() < deadline:
            remaining_budget = max_calls - used_calls
            if remaining_budget > 20:
                try:
                    result = opt.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining_budget,
                            'maxfev': remaining_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (RuntimeError, Exception):
                    pass
        
        # If no valid best_x was found, try a single random point
        if best_x is None:
            try:
                x_rand = np.random.uniform(-5, 5, self.dim)
                fp = safe_func(x_rand)
                best_f = fp
                best_x = x_rand.copy()
            except:
                pass
        
        return best_x, best_f
