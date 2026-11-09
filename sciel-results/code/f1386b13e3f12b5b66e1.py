import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given by three points."""
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def _min_triangle_area(points):
    """Find the minimum area among all triangles formed by the points."""
    n = len(points)
    min_area = float('inf')
    # For n=13, there are C(13,3) = 286 triangles. This is fast enough to compute directly.
    for i, j, k in combinations(range(n), 3):
        area = _triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # If hull calculation fails (e.g., collinear points), return a small value to penalize
        return 1e-10

def _ratio(points):
    """Calculate the score: min triangle area / hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area == 0:
        return 0.0
    return min_area / hull_area

def _neg_ratio(flat_points):
    """Objective function for the optimizer (minimize negative ratio)."""
    points = flat_points.reshape(-1, 2)
    return -_ratio(points)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 300  # Use 300 seconds for optimization to be safe
    
    best_points = None
    best_score = -float('inf')
    
    # Strategy 1: Start with a regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Strategy 2: Start with 7 outer points (regular heptagon) and 6 inner points (regular hexagon scaled down)
    # This topology is often optimal for n=13
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_pts = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    init_points_2 = np.vstack([outer_pts, inner_pts])

    initializations = [init_points, init_points_2]
    
    for init in initializations:
        if time.time() - start_time > time_limit:
            break
            
        # Flatten points for the optimizer
        x0 = init.flatten()
        
        try:
            result = minimize(
                _neg_ratio, 
                x0, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.success or result.fun < -best_score:
                current_points = result.x.reshape(-1, 2)
                current_score = _ratio(current_points)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points
                    
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue
            
    # If optimization didn't yield a good result, fallback to the best initial
    if best_points is None:
        best_points = init_points
    else:
        # Final check: ensure points are valid (finite)
        if not np.all(np.isfinite(best_points)):
            best_points = init_points
            
    return best_points

# EVOLVE-BLOCK-END
