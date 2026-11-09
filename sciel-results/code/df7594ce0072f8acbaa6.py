"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start optimizer with grid exploration and (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration
        # Create a grid of candidate starting points to hit different peaks.
        # For 5D, a 2^5 = 32 grid is a good start.
        # We evaluate the center of each grid cell.
        grid_size = 2
        coords = np.linspace(-5, 5, grid_size + 1)
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        for m in range(grid_size):
                            p = np.array([
                                (coords[i] + coords[i+1]) / 2,
                                (coords[j] + coords[j+1]) / 2,
                                (coords[k] + coords[k+1]) / 2,
                                (coords[l] + coords[l+1]) / 2,
                                (coords[m] + coords[m+1]) / 2
                            ])
                            grid_points.append(p)
        
        # Shuffle to avoid bias
        rng.shuffle(grid_points)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        remaining_budget = self.budget

        # Evaluate grid points
        for p in grid_points:
            if calls_made >= remaining_budget:
                break
            if time.time() > deadline:
                break
            
            try:
                fp = func(p)
                calls_made += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except Exception:
                continue
        
        # If we didn't make any calls, we are in trouble, but the contract says 
        # "exception before the first call makes the whole result invalid".
        # We assume func works.
        
        if best_x is None:
             # Fallback if grid points failed or budget was 0
             best_x = rng.uniform(-5, 5, self.dim)
             try:
                 best_f = func(best_x)
                 calls_made += 1
             except:
                 return best_x, np.inf

        # Phase 2: Local Refinement
        # Use the best point found as the seed for (1+1)-ES
        x = best_x
        fx = best_f
        sigma = 1.0  # Start with a smaller sigma since we are likely near a peak
        
        # We have limited budget left
        # Allocate remaining budget to refinement
        # A simple (1+1)-ES is fast per step.
        
        while calls_made < remaining_budget:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
                calls_made += 1
            except Exception:
                break
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1  # Slightly increase step size on success
            else:
                sigma *= 0.9  # Decrease step size on failure
                
            # If sigma becomes too small, we've converged
            if sigma < 1e-6:
                break

        return x, fx

# EVOLVE-BLOCK-END
