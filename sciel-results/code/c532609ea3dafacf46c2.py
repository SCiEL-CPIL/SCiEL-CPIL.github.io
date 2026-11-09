import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin

# Precompute triangle indices for n=13
N_POINTS = 13
TRIANGLES = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    # points shape: (13, 2)
    # TRIANGLES shape: (286, 3)
    
    p1 = points[TRIANGLES[:, 0]]
    p2 = points[TRIANGLES[:, 1]]
    p3 = points[TRIANGLES[:, 2]]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    # In 2D, cross product is a scalar: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If hull fails (e.g., all collinear), return a very small number
        # to penalize this configuration heavily in the objective
        return 1e-10

def _objective(params):
    """
    Objective function to minimize: - (A_min / A_hull)
    """
    points = params.reshape(N_POINTS, 2)
    
    # Check for invalid points (NaN, Inf)
    if not np.all(np.isfinite(points)):
        return 1e10
        
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-9:
        return 1e10
        
    ratio = a_min / a_hull
    return -ratio

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region 
    to maximize the area of the smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    best_points = None
    best_score = -np.inf
    
    # Pre-allocate some initializations
    rng = np.random.default_rng(seed=12345)
    
    # List of initial point sets to try
    initializations = []
    
    # 1. Random uniform in square
    initializations.append(rng.random((N_POINTS, 2)))
    
    # 2. Points on a circle (Heilbronn points often lie on the boundary)
    angles = np.linspace(0, 2*np.pi, N_POINTS, endpoint=False) + rng.random(N_POINTS)*0.1
    radii = np.ones(N_POINTS)
    initializations.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))
    
    # 3. Grid with jitter
    grid = np.array(list(itertools.product(np.linspace(0,1,4), np.linspace(0,1,4))))[:13]
    initializations.append(grid + rng.normal(0, 0.05, (13,2)))
    
    # 4. Another random
    initializations.append(rng.random((N_POINTS, 2)))
    
    # 5. Another circle with different radii
    angles2 = np.linspace(0, 2*np.pi, N_POINTS, endpoint=False)
    radii2 = np.ones(N_POINTS) * 0.9
    initializations.append(np.column_stack([np.cos(angles2)*radii2, np.sin(angles2)*radii2]))

    # We will perform local optimization from each initialization
    # And potentially restart from the best found so far with perturbations
    
    num_restarts = 50 # We can do many because Nelder-Mead on 26 vars is fast enough
    
    for i in range(num_restarts):
        if time.time() > _DEADLINE:
            break
            
        # Choose an initialization
        if i < len(initializations):
            init_points = initializations[i].copy()
        else:
            # Perturb the best found so far
            if best_points is not None:
                init_points = best_points + rng.normal(0, 0.05, (N_POINTS, 2))
            else:
                init_points = rng.random((N_POINTS, 2))
        
        # Flatten for optimizer
        x0 = init_points.flatten()
        
        # Nelder-Mead settings
        # Initial simplex size should be small relative to the problem scale
        # Since we are in [0,1] roughly, 0.1 is a reasonable step
        options = {
            'maxiter': 2000,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options=options)
            
            if res.success or res.fun < -best_score:
                # Check if result is valid
                pts = res.x.reshape(N_POINTS, 2)
                if np.all(np.isfinite(pts)):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except Exception as e:
            pass

    # If no optimization succeeded, return a valid random set
    if best_points is None:
        best_points = rng.random((N_POINTS, 2))
        
    return best_points
