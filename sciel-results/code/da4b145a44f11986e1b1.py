# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _compute_ratio(points: np.ndarray) -> float:
    """
    Compute R = (d_min / d_max)^2 for the given points.
    """
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(points_flat: np.ndarray) -> float:
    """
    Objective function to minimize.
    We want to maximize (d_min / d_max), so we minimize its negative.
    """
    n = len(points_flat) // 2
    points = points_flat.reshape((n, 2))
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        return 1.0 # Large penalty
        
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    
    if d_max == 0:
        return 0.0
        
    return -(d_min / d_max)

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Initial configurations
    inits = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    inits.append(grid_points)
    
    # 2. Regular 16-gon (Circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)
    
    # 3. Random
    np.random.seed(42)
    rand_points = np.random.randn(n, d)
    inits.append(rand_points)
    
    # 4. Slightly perturbed grid
    perturbed_grid = grid_points + 0.01 * np.random.randn(n, d)
    inits.append(perturbed_grid)

    best_score = -1.0
    best_points = inits[0] # Fallback
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        # Use Nelder-Mead
        try:
            # Adaptive options for robustness
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
            
            # Check result
            if np.all(np.isfinite(res.x)):
                points = res.x.reshape((n, 2))
                # Calculate the actual ratio to compare
                d = pdist(points)
                d_min = np.min(d)
                d_max = np.max(d)
                if d_max > 0:
                    score = d_min / d_max
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception:
            pass
            
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
