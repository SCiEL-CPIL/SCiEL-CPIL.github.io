"""Combined optimizer: multi-start global sampling + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    
    Strategy:
    1. Phase 1: Global exploration using a mix of random and quasi-random points.
       This quickly identifies promising regions in the shifted/rotated landscape.
    2. Phase 2: Local refinement using Nelder-Mead simplex method starting from
       the best points found in Phase 1.
    
    This approach avoids the slow convergence of population-based methods (like CMA-ES)
    in early iterations and exploits the multimodal nature of f17 by jumping to local
    basins via global sampling, then refining with efficient local search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 290s to be safe
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        # Track state
        best_x = None
        best_f = np.inf
        calls = 0
        found_points = []  # Store (f, x) pairs for potential restarts
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)
        
        # Evaluate function wrapper
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x_clipped = clip(x)
            try:
                fx = func(x_clipped)
            except:
                return np.inf
            
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            
            # Store point for potential restarts
            if np.isfinite(fx):
                found_points.append((fx, x_clipped.copy()))
                
            return fx

        # Phase 1: Global Exploration
        # Use a mix of random and stratified sampling
        # We'll spend about 40% of budget on global exploration
        global_budget = max(20, int(budget * 0.4))
        
        # Generate initial points
        # 1. Center of the box
        center = np.zeros(n)
        f_center = evaluate(center)
        
        # 2. Random points
        rng = np.random.default_rng(42)
        num_random = global_budget - 1
        
        # Use stratified sampling for better coverage
        # Divide the box into sub-regions and sample one point per region
        # For n=5, we can do a simple grid or random stratification
        # Let's use a combination: some uniform random, some from a coarse grid
        
        # Coarse grid points (corners and midpoints)
        grid_points = []
        # Corners
        for i in range(2**n):
            point = np.array([((i >> j) & 1) * 10 - 5 for j in range(n)])
            grid_points.append(point)
        
        # Midpoints of edges
        for i in range(n):
            for j in range(i+1, n):
                point = np.zeros(n)
                point[i] = 0
                point[j] = 0
                # Other coordinates random
                point = rng.uniform(-5, 5, n)
                point[i] = 0
                point[j] = 0
                grid_points.append(point)
        
        # Sample from grid and random
        for _ in range(num_random):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Alternate between grid and random
            if len(grid_points) > 0 and rng.random() < 0.3:
                # Pick a random grid point
                idx = rng.integers(0, len(grid_points))
                x = grid_points[idx]
                grid_points.pop(idx)
            else:
                x = rng.uniform(-5, 5, n)
            
            evaluate(x)
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Start from the best points found in Phase 1
        # Sort found_points by fitness
        found_points.sort(key=lambda x: x[0])
        
        # Try Nelder-Mead from the top few points
        num_restarts = min(5, len(found_points))
        
        for i in range(num_restarts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            start_x = found_points[i][1]
            
            # Define objective for Nelder-Mead
            def objective(x):
                if calls >= budget:
                    return np.inf
                if time.time() > deadline:
                    return np.inf
                return evaluate(x)
            
            # Nelder-Mead options
            # maxiter should be limited to conserve budget
            # maxfev is the key constraint
            remaining_budget = budget - calls
            if remaining_budget < 10:
                break
            
            # Use a fraction of remaining budget for this restart
            # First restart gets more, subsequent ones get less
            if i == 0:
                maxfev = min(remaining_budget, 100)
            else:
                maxfev = min(remaining_budget, 50)
            
            try:
                result = minimize(
                    objective, 
                    start_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'maxfev': maxfev,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # Update best if Nelder-Mead found something better
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = result.fun
                    best_x = clip(result.x)
                    
            except Exception:
                pass
        
        # Phase 3: Final Polish
        # If we have budget left, do a fine-grained local search around the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Simple coordinate descent or small random perturbations
                for _ in range(min(remaining, 20)):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Small random step
                    step_size = 0.1
                    x_perturbed = best_x + step_size * rng.standard_normal(n)
                    x_perturbed = clip(x_perturbed)
                    f_perturbed = evaluate(x_perturbed)
                    
                    if f_perturbed < best_f:
                        best_x = x_perturbed
                        best_f = f_perturbed
        
        return best_x, best_f
# EVOLVE-BLOCK-END
