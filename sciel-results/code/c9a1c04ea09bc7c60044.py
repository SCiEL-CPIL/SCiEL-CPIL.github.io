"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _generate_points(params):
    """
    Generate 8 points from two regular squares.
    params: [r1, theta1, r2, theta2, x2, y2]
    Square 1 is centered at (0,0).
    Square 2 is centered at (x2, y2).
    """
    r1, theta1, r2, theta2, x2, y2 = params
    if r1 < 0 or r2 < 0:
        return None

    # Square 1: radius r1, orientation theta1, center (0,0)
    angles1 = theta1 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2: radius r2, orientation theta2, center (x2, y2)
    angles2 = theta2 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts2 = np.column_stack([r2 * np.cos(angles2) + x2, r2 * np.sin(angles2) + y2])
    
    return np.vstack([pts1, pts2])

def _score(params):
    """Compute negative log of the score (to minimize). Returns large value if invalid."""
    pts = _generate_points(params)
    if pts is None:
        return 1e10
    
    # Check distinctness and compute distances
    dists = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == j:
                dists[i, j] = 1.0 # Placeholder, won't be used in log
            else:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-9:
                    return 1e10
                dists[i, j] = d
    
    # Compute diameter
    max_dist = np.max(dists)
    if max_dist < 1e-12:
        return 1e10
    
    # Scale to diameter 2
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    # Recompute distances after scaling (or just scale the dists)
    # dists_scaled = dists * scale
    # log_product = sum(log(dists_scaled[i,j]) for i!=j)
    
    log_product = 0.0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d = dists[i, j] * scale
            if d < 1e-15:
                return 1e10
            log_product += np.log(d)
    
    n = 8
    log_score = log_product - n * np.log(n)
    
    return -log_score

def construct():
    # Parameters: r1, theta1, r2, theta2, x2, y2
    # Bounds
    bounds = [
        (0.1, 2.0),  # r1
        (0.0, np.pi / 2),  # theta1
        (0.1, 2.0),  # r2
        (0.0, np.pi / 2),  # theta2
        (-2.0, 2.0),  # x2
        (-2.0, 2.0),  # y2
    ]
    
    # Initial guess: regular octagon
    # r1=1, theta1=0, r2=1, theta2=pi/4, x2=0, y2=0
    x0 = [1.0, 0.0, 1.0, np.pi/4, 0.0, 0.0]
    
    # Use differential evolution for global search
    result = differential_evolution(
        _score,
        bounds,
        seed=42,
        maxiter=2000,
        tol=1e-10,
        polish=True,
        workers=1
    )
    
    if result.success or result.fun < 1e9:
        best_params = result.x
    else:
        best_params = x0
    
    # Local refinement with L-BFGS-B or similar if needed, but DE polish should suffice
    # Let's try a quick local polish on the best found
    try:
        res_local = minimize(_score, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        if res_local.fun < _score(best_params):
            best_params = res_local.x
    except Exception:
        pass

    pts = _generate_points(best_params)
    
    if pts is None:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
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
