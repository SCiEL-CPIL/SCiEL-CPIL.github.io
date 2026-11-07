"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Global Grid Search followed by Local Refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 290
        
        # Phase 1: Global Grid Search
        # We need to cover [-5, 5]^5. 
        # Let's allocate ~80% of the budget to the global phase.
        global_budget = int(self.budget * 0.8)
        
        # Determine grid resolution.
        # If we use N points per dimension, total points = N^dim.
        # For dim=5, N=4 gives 1024 points. N=3 gives 243 points.
        # We have 400 budget. Let's aim for a grid that fits.
        # 5^5 = 3125 (too many). 4^5 = 1024 (too many). 3^5 = 243 (fits).
        # Let's use a 3-point grid per dimension initially, then refine?
        # Or a random subset of a finer grid?
        
        # Strategy: Generate a set of candidate points that cover the space.
        # A 3x3x3x3x3 grid = 243 points.
        # We have 400 points. We can afford a 4x4x4x4x4 grid? No, 1024.
        # How about a 3x3x3x3x4 grid? 324 points.
        # Let's just use a random sampling that is stratified to ensure coverage, 
        # or a simple grid if it fits.
        
        # Let's use a grid of 3 points per dimension (low, mid, high).
        # Points: -5, 0, 5.
        # This is 3^5 = 243 evaluations.
        
        grid_points = []
        # Create 3 levels: -5, 0, 5
        levels = [-5.0, 0.0, 5.0]
        
        # Generate all combinations
        import itertools
        for combo in itertools.product(levels, repeat=self.dim):
            grid_points.append(np.array(combo))
            
        # We have 243 points. Budget is 400 for this phase.
        # We can add some random points to fill the gap and increase resolution.
        # Or we can use a slightly finer grid for some dimensions?
        # Let's stick to the 3-level grid first.
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Evaluate grid points
        for x in grid_points:
            if time.time() > deadline:
                break
            if calls_made >= global_budget:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local Refinement
        # Use the remaining budget to refine the best point found.
        # We can use Nelder-Mead or a simple local random walk.
        # Nelder-Mead is deterministic and fast for 5D.
        # However, we must count function calls.
        # scipy.optimize.minimize doesn't easily expose a strict call counter 
        # that stops exactly at a budget limit without wrapping.
        # Let's implement a simple local hill-climbing/random-walk refinement 
        # to strictly control the budget.
        
        remaining_budget = self.budget - calls_made
        
        if best_x is not None and remaining_budget > 0:
            # Simple Local Search: Random walk with decreasing step size
            x = best_x
            fx = best_f
            sigma = 1.0
            
            # Use a fixed number of steps or until budget/time runs out
            # Let's use a (1+1)-ES style refinement but with smaller sigma
            while calls_made < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x, fx
                    sigma *= 1.1 # Slight expansion if improving
                else:
                    sigma *= 0.9 # Shrink if not improving
                    
                # If sigma becomes too small, stop
                if sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
