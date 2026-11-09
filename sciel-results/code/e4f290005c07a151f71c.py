"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Adaptive lattice refinement (deterministic) with local Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit
        calls = 0

        def fcall(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            calls += 1
            return val

        # Phase 1: Adaptive lattice refinement (deterministic)
        # Start with a coarse grid, then iteratively refine the best cell
        # Use a priority queue to always split the cell with the best center value
        
        # Initial grid: 3^dim cells (manageable for dim=5: 243 cells)
        # But 243 calls is a lot. Let's use 2^dim = 32 cells first.
        # Actually, let's start with the center and expand.
        
        # Better approach: 
        # 1. Evaluate center (0,0,0,0,0)
        # 2. Evaluate a set of initial points to get a sense of the landscape
        # 3. Use a priority queue of hyper-rectangles, split the best one
        
        # Initial points: center + axis-aligned points
        init_points = [np.zeros(dim)]
        for d in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[d] = sign * 2.0
                init_points.append(p)
        
        # Evaluate initial points
        best_f = float('inf')
        best_x = None
        for p in init_points:
            f = fcall(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
        
        # Now use adaptive refinement
        # Each cell is defined by (center, half_widths)
        # Priority: lower f(center) is better (we want to refine promising regions)
        # We'll maintain a max-heap (using negative values) of cells to refine
        
        # Start with the cell containing the best point
        # Actually, let's define cells more carefully.
        # A cell is a hyper-rectangle. We split it by halving its longest edge.
        
        # Initialize with one large cell covering the whole space, centered at best_x
        # But we don't know the extent. Let's use the known box [-5,5]^dim.
        
        # Strategy:
        # - Maintain a list of "active" cells to refine
        # - Start with the whole box
        # - Each iteration: evaluate the center of the best active cell, 
        #   then split it into two by halving the longest dimension
        # - Keep the better child cell in the active set
        
        # This is essentially a deterministic version of adaptive random search.
        
        # Let's use a simpler approach: 
        # 1. Coarse grid search (2^dim = 32 points)
        # 2. Local refinement around the best point using Nelder-Mead
        
        # Phase 1: Coarse grid
        # 2^dim points in [-5,5]^dim
        grid_points = []
        for mask in range(2 ** dim):
            p = np.zeros(dim)
            for d in range(dim):
                if mask & (1 << d):
                    p[d] = 5.0
                else:
                    p[d] = -5.0
            grid_points.append(p)
        
        # That's 32 points. Let's also add the center.
        # Actually, let's use a smarter grid: 
        # For dim=5, we can afford a 3^dim = 243 grid? No, that's too many.
        # Let's use 2^dim + 1 (center) = 33 points.
        
        # Evaluate all grid points
        for p in grid_points:
            f = fcall(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()
        
        # Phase 2: Local refinement using Nelder-Mead
        # Use scipy.optimize.minimize with Nelder-Mead
        from scipy.optimize import minimize
        
        # Set up Nelder-Mead
        # Initial simplex: centered at best_x
        x0 = best_x.copy()
        
        # Use a reasonable initial step size
        # The grid spacing is 10, so let's start with step size 1.0
        options = {
            'maxiter': 50,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        # We need to track calls and respect the budget
        # scipy's minimize will call func multiple times.
        # We'll wrap it to count calls and stop if we exceed budget.
        
        nm_calls = [0]
        def nm_func(x):
            nonlocal deadline
            if time.time() > deadline or calls >= budget:
                return best_f  # Return best known to stop
            nm_calls[0] += 1
            return fcall(x)
        
        try:
            result = minimize(nm_func, x0, method='Nelder-Mead', options=options)
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except:
            pass
        
        # If we still have budget, do another round of refinement
        # Try multiple starting points for Nelder-Mead
        if calls < budget - 50:
            # Try starting from a few nearby points
            step = 0.5
            for d in range(dim):
                for sign in [1, -1]:
                    if calls >= budget - 10 or time.time() > deadline:
                        break
                    p = best_x.copy()
                    p[d] += sign * step
                    p = np.clip(p, -5, 5)
                    f = fcall(p)
                    if f < best_f:
                        best_f = f
                        best_x = p.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
