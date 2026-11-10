import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def compute_ratio(points):
    """
    Computes the ratio A_min / A_hull for a given set of points.
    Optimized for points in angular order (convex position).
    """
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    n = points.shape[0]
    
    # Calculate Convex Hull Area
    # If points are in convex position and ordered by angle, 
    # the convex hull is the polygon formed by the points.
    # We check convexity first.
    
    # Check if polygon is convex
    # Cross product of consecutive edges should have the same sign
    convex = True
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # We assume counter-clockwise ordering, so cross should be positive
        # If it's negative or zero, it's not strictly convex or concave
        if cross <= 1e-12:
            convex = False
            break
            
    if convex:
        # Shoelace formula for area
        x = points[:, 0]
        y = points[:, 1]
        # x_i * y_{i+1} - x_{i+1} * y_i
        a_hull = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if a_hull <= 1e-10:
            return -np.inf
    else:
        # Fallback to ConvexHull if not convex
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull <= 1e-10:
                return -np.inf
        except Exception:
            return -np.inf

    # Calculate all triangle areas
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    a_min = np.min(areas)
    
    if a_min <= 1e-10:
        return -np.inf
        
    return a_min / a_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    def unpack(params):
        r = params
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        
    def objective(params):
        points = unpack(params)
        ratio = compute_ratio(points)
        return -ratio

    # Bounds for radii
    # Keep points near the unit circle to maintain convexity and reasonable scale
    bounds = [(0.5, 1.5)] * n
    
    # Use differential evolution for global optimization
    # popsize=10, maxiter=500 is a reasonable balance for 13 dimensions
    # seed for reproducibility if needed, but default is fine
    result = differential_evolution(objective, bounds, maxiter=300, popsize=10, seed=42, polish=True)
    
    best_params = result.x
    best_points = unpack(best_params)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
