"""Model-Based Local Search with Adaptive Error Repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Model-Based Local Search Optimizer.
    
    Strategy:
    1. Uses a local quadratic approximation (via finite differences or simplex) 
       to guide steps, assuming local smoothness.
    2. Implements 'error_repair': if a proposed step fails to improve fitness,
       the local model is invalidated, the search radius is adapted, and a 
       randomized restart or larger step is taken.
    3. Hierarchical radii: Starts wide, narrows upon improvement, widens upon failure.
    
    This avoids the overhead of large populations and focuses budget on 
    high-quality local exploitation with robust escape mechanisms.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin

        x_best = np.random.uniform(-5, 5, dim)
        f_best = float('inf')
        calls = 0
        best_x = x_best.copy()
        best_f = float('inf')
        
        # Adaptive search radius
        radius = 4.0  # Start wide
        min_radius = 1e-4
        max_radius = 9.0
        
        # Simplex for local gradient estimation (Nelder-Mead style but adaptive)
        # We will use a "Model-Free" local search that mimics SQP steps:
        # Estimate gradient via central differences, then take a step along negative gradient
        # scaled by an adaptive learning rate. If it fails, we "repair" by randomizing.
        
        # Alternative: Pure Nelder-Mead is robust but slow. 
        # Let's use a "Pattern Search" variant with adaptive step sizes per dimension.
        # This is more robust to noise and non-smoothness (Rastrigin) than pure gradient.
        
        # State: current point, step sizes per dimension
        x_curr = np.random.uniform(-5, 5, dim)
        f_curr = func(x_curr)
        calls += 1
        
        if f_curr < best_f:
            best_f = f_curr
            best_x = x_curr.copy()

        # Step sizes: start with large steps
        steps = np.ones(dim) * 2.0
        
        # Error repair counter
        repair_count = 0
        max_repair_before_reset = 10

        # Main loop
        while calls < budget and time.time() < deadline:
            
            # Check for stagnation: if no improvement for a while, trigger global restart
            # We track iterations without improvement
            # For simplicity, we use the 'repair_count' as a proxy for local failure.
            
            # 1. Generate candidate steps using pattern search logic
            # Try expanding/contracting along each axis
            
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive step
                x_try = x_curr.copy()
                x_try[i] += steps[i]
                x_try = np.clip(x_try, -5, 5)
                
                f_try = func(x_try)
                calls += 1
                
                if f_try < f_curr:
                    # Improvement: move to new point
                    x_curr = x_try
                    f_curr = f_try
                    improved = True
                    
                    # Update global best
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                    
                    # Expand step size for this dimension (success)
                    steps[i] *= 1.5
                    if steps[i] > max_radius:
                        steps[i] = max_radius
                    # Reset repair count on improvement
                    repair_count = 0
                    break # Restart loop from dimension 0 to keep pattern search consistent
                    
                # Try negative step
                x_try = x_curr.copy()
                x_try[i] -= steps[i]
                x_try = np.clip(x_try, -5, 5)
                
                f_try = func(x_try)
                calls += 1
                
                if f_try < f_curr:
                    # Improvement
                    x_curr = x_try
                    f_curr = f_try
                    improved = True
                    
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                    
                    # Expand step size
                    steps[i] *= 1.5
                    if steps[i] > max_radius:
                        steps[i] = max_radius
                    repair_count = 0
                    break # Restart loop
                    
            if improved:
                continue
            
            # 2. No improvement found in this pass: Error Repair
            repair_count += 1
            
            # Shrink all step sizes
            steps *= 0.5
            steps = np.maximum(steps, min_radius)
            
            # If we have repaired too many times without improvement, 
            # the local model is likely stuck in a bad basin or flat area.
            # Trigger a "Global Repair": jump to a new random location 
            # or around the global best.
            if repair_count > max_repair_before_reset:
                # Decide where to jump
                # 50% chance to jump near global best (exploitation)
                # 50% chance to jump randomly (exploration)
                if np.random.rand() < 0.5:
                    # Jump near global best with some noise
                    noise_scale = 1.0 + np.random.rand() * 2.0
                    x_curr = best_x + np.random.normal(0, noise_scale, dim)
                else:
                    # Random restart
                    x_curr = np.random.uniform(-5, 5, dim)
                
                x_curr = np.clip(x_curr, -5, 5)
                f_curr = func(x_curr)
                calls += 1
                
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr.copy()
                
                # Reset step sizes to medium
                steps = np.ones(dim) * 1.0
                repair_count = 0
                
                # If we are very close to the end of budget, stop restarting
                if budget - calls < 20:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
