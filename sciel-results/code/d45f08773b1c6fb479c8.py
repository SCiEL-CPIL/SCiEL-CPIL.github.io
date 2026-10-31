"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Variables: x_0, y_0, ..., x_20, y_20, h
    # Total 43 variables
    # w = 2 - h
    
    def objective(params):
        if time.time() > deadline:
            return 0.0 # Or a large penalty, but we want to stop
            
        coords = params[:42].reshape(N, 2)
        h = params[42]
        w = 2.0 - h
        
        # Check if h is valid (positive)
        if h <= 0 or w <= 0:
            return 0.0
            
        # Calculate radii
        radii = np.zeros(N)
        for i in range(N):
            x, y = coords[i]
            # Distance to walls
            min_dist = min(x, w - x, y, h - y)
            
            # Distance to other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x - coords[j][0]
                dy = y - coords[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
                
            radii[i] = 0.5 * min_dist
            
        return -np.sum(radii) # We minimize negative sum
    
    # Initial bounds
    # x, y in [0, 2] is a safe bound for any aspect ratio with perimeter 4
    # h in [0.5, 1.5] covers square and reasonable rectangles
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0)) # x_i
        bounds.append((0.0, 2.0)) # y_i
    bounds.append((0.5, 1.5)) # h
    
    # Run differential evolution
    # maxiter and popsize control the effort. 
    # We can run multiple short runs or one long run.
    # DE is stochastic, so we can just run it until time runs out or it converges.
    
    best_result = None
    best_score = -np.inf
    
    # We will run DE in a loop, checking time.
    # DE itself doesn't easily support a "stop after X seconds" callback in all versions,
    # but we can use `callback` or just run with a fixed number of iterations and check time.
    # Better: run a single DE with a reasonable number of iterations, or multiple restarts.
    
    # Let's do a single run with a generous budget, or multiple shorter runs.
    # Given 110s, we can do a few runs.
    
    # Strategy: Run DE with maxiter=50, popsize=15. This is fast.
    # Then repeat if time remains.
    
    while time.time() < deadline:
        # Use a seed to ensure reproducibility if we wanted, but random is fine for search
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-6, 
                seed=42 # Fixed seed for this block to be deterministic? Or random? 
                # Let's use random seed for diversity in case we run multiple times
                # But for a single submission, fixed is fine.
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract best coordinates and radii
    coords = best_result[:42].reshape(N, 2)
    h = best_result[42]
    w = 2.0 - h
    
    # Recalculate radii for the final output
    radii = np.zeros(N)
    for i in range(N):
        x, y = coords[i]
        min_dist = min(x, w - x, y, h - y)
        for j in range(N):
            if i == j:
                continue
            dx = x - coords[j][0]
            dy = y - coords[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, dist)
        radii[i] = 0.5 * min_dist
        
    # Construct output array (x, y, r)
    result = np.column_stack((coords, radii))
    
    return result

# EVOLVE-BLOCK-END
