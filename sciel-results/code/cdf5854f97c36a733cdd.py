"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _generate_points(r1, r2, theta1, theta2, dy):
    """Generate 8 points: two concentric regular squares with independent radii, orientations, and vertical offset."""
    # Square 1: radius r1, orientation theta1
    angles1 = theta1 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2: radius r2, orientation theta2, vertical offset dy
    angles2 = theta2 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2) + dy])
    
    return np.vstack([pts1, pts2])

def _score(params):
    """Compute negative log of the score (to minimize). Returns large value if invalid."""
    r1, r2, theta1, theta2, dy = params
    if r1 < 0 or r2 < 0:
        return 1e10
    
    pts = _generate_points(r1, r2, theta1, theta2, dy)
    
    # Check distinctness
    min_dist = np.inf
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e10
            min_dist = min(min_dist, d)
    
    # Compute diameter
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            max_dist = max(max_dist, d)
    
    if max_dist < 1e-12:
        return 1e10
    
    # Scale to diameter 2
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    # Compute product of distances over ordered pairs
    # Ordered pairs: i != j, so 8*7 = 56 pairs
    log_product = 0.0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
            if d < 1e-15:
                return 1e10
            log_product += np.log(d)
    
    # Score = Delta / n^n where Delta is product of distances
    # log(score) = log_product - n * log(n)
    n = 8
    log_score = log_product - n * np.log(n)
    
    # We want to maximize score, so minimize -log_score
    return -log_score

def construct():
    # Define bounds for the 5 parameters
    bounds = [
        (0.1, 2.0),  # r1
        (0.1, 2.0),  # r2
        (0.0, np.pi / 2),  # theta1
        (0.0, np.pi / 2),  # theta2
        (-1.0, 1.0),  # dy
    ]
    
    # Regular octagon parameters
    regular_params = [1.0, 1.0, 0.0, np.pi / 4, 0.0]
    best_score_val = _score(regular_params)
    best_params = regular_params
    
    # Multistart optimization using L-BFGS-B
    # We will try several random starting points to escape local minima
    np.random.seed(42)
    num_starts = 50
    
    for _ in range(num_starts):
        # Generate a random starting point within bounds
        start = np.array([
            np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(5)
        ])
        
        # Run local optimization
        res = minimize(
            _score, 
            start, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.fun < best_score_val:
            best_score_val = res.fun
            best_params = res.x
            
    pts = _generate_points(*best_params)
    
    # Scale to diameter 2
    max_dist = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.linalg.norm(pts[i] - pts[j])
            max_dist = max(max_dist, d)
    
    if max_dist < 1e-12:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    return [[float(pts_scaled[i][0]), float(pts_scaled[i][1])] for i in range(8)]
# EVOLVE-BLOCK-END
