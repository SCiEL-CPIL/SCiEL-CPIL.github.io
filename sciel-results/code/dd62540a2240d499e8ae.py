"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global grid exploration followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        # Phase 1: Global Grid Exploration
        # We use a coarse grid to find the general region of the minimum.
        # In 5D, 5 points per dimension = 3125 points. This is too many for 500 budget.
        # Let's use 3 points per dimension: 3^5 = 243 points.
        # This leaves ~257 calls for refinement.
        
        # Grid levels: -3, 0, 3 (covering most of the [-5,5] range where minima likely are)
        # Actually, the problem states minimizer is in [-4, 4]^5.
        # Let's use 3 levels: -4, 0, 4.
        grid_levels = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        import itertools
        grid_points = list(itertools.product(grid_levels, repeat=self.dim))
        
        # Shuffle to ensure we don't always start from the same corner if there's bias, 
        # but for a grid, order matters less for coverage. 
        # However, to maximize early AOCC, we should probably evaluate the center first?
        # No, random shuffle is safer to avoid systematic bias if the function is weird.
        # But wait, the grid is deterministic. Let's just iterate.
        # Actually, let's randomize the order of the grid points to be safe against 
        # potential correlations in the hidden shift/rotation? 
        # No, the grid is symmetric. 
        # Let's just evaluate them.
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Evaluate Grid
        for pt in grid_points:
            if time.time() > deadline:
                break
            x = np.array(pt)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Local Refinement
        # Use a (1+1)-ES starting from the best grid point.
        # Budget remaining
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 0:
            x = best_x.copy()
            fx = best_f
            # Start with a smaller sigma since we are already in a promising region
            sigma = 1.0
            
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(0).standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1  # Slight increase
                else:
                    sigma *= 0.8  # Decrease
                    if sigma < 1e-6:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
