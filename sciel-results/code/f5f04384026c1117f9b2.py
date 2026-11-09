# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # Leave 20s margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If hull fails (e.g., collinear points), return a large penalty or 0
        # Returning 0 would make the ratio infinite if A_min > 0, which is bad for minimization of -ratio.
        # If A_hull is 0, the configuration is invalid.
        return 1e-10

def _score(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return -1e10  # Invalid configuration
        
    return a_min / a_hull

def _objective(params):
    """
    Objective function for minimization.
    We want to maximize the score, so we minimize the negative score.
    """
    points = params.reshape(13, 2)
    s = _score(points)
    return -s

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points_init = np.column_stack((x, y))
    
    best_params = points_init.flatten()
    best_score = _score(points_init)
    
    # Optimization
    try:
        res = minimize(_objective, best_params, method='Powell', options={'maxiter': 2000, 'maxfev': 50000})
        
        if res.success or res.fun < -best_score:
            best_params = res.x
            best_score = -res.fun
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE:
        return best_params.reshape(13, 2)
    
    # Try a few more restarts if time permits
    # The regular polygon is usually the global optimum for small n, but let's try a few random perturbations
    rng = np.random.default_rng(seed=42)
    
    for i in range(10):
        if time.time() > _DEADLINE - 60:
            break
            
        # Perturb the initial regular polygon
        noise = rng.normal(0, 0.1, (n, 2))
        points_pert = points_init + noise
        params_pert = points_pert.flatten()
        
        try:
            res = minimize(_objective, params_pert, method='Powell', options={'maxiter': 1000, 'maxfev': 20000})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass
            
    return best_params.reshape(13, 2)

# EVOLVE-BLOCK-END
