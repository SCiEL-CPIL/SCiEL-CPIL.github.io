# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    # For n=13, C(13,3) = 286. This is small enough to generate all at once.
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Vectorized cross product magnitude / 2
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1x*v2y - v1y*v2x
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
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    Returns 0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
        
    return a_min / a_hull

def _objective(points_flat):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes negative score.
    """
    n = 13
    points = points_flat.reshape((n, 2))
    s = _score(points)
    if s == 0:
        return 1e10 # Large penalty for invalid/degenerate
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = 0.0
    
    # Strategy: Try multiple initializations and optimize.
    # 1. Regular polygon
    # 2. Random points
    # 3. Grid-like
    
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_poly.flatten())
    
    # 2. Random points in unit square
    rng = np.random.default_rng(seed=42)
    for i in range(5):
        rand_pts = rng.random((n, 2))
        initializations.append(rand_pts.flatten())
        
    # 3. Points on a circle with some perturbation
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.random(n) * 0.1
        radii = 1.0 + rng.random(n) * 0.1
        circ_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(circ_pts.flatten())

    # Evaluate initializations first
    for init in initializations:
        pts = init.reshape((n, 2))
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Optimize from the best initializations
    # Use Nelder-Mead as it doesn't require gradients and is robust for small dimensions
    # Dimension is 26. Nelder-Mead is O(N^2) per step? Might be slow.
    # Let's try Powell or Nelder-Mead with limited iterations.
    
    # Sort initializations by score
    init_scores = []
    for init in initializations:
        pts = init.reshape((n, 2))
        s = _score(pts)
        init_scores.append((s, init))
    init_scores.sort(reverse=True)
    
    # Take top 3 initializations for optimization
    for s, init in init_scores[:3]:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(_objective, init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if time.time() > _DEADLINE - 5:
                 # If we are close to deadline, don't evaluate if it might hang? 
                 # Minimize returns quickly if it hits maxiter.
                 pass

            opt_pts = res.x.reshape((n, 2))
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
                
        except Exception:
            continue
            
    # If no valid points found (should not happen), return regular polygon
    if best_points is None:
        best_points = reg_poly
        
    return best_points

# EVOLVE-BLOCK-END
