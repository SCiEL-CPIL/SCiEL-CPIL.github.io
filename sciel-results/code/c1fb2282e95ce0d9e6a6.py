import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def compute_ratio(points):
    """
    Computes the ratio A_min / A_hull for a given set of points.
    """
    # Check for invalid points
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    n = points.shape[0]
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 1e-10:
            return -np.inf
    except Exception:
        return -np.inf

    # Calculate all triangle areas
    # Indices for all combinations of 3 points
    i, j, k = np.indices((n, n, n))
    # Only keep i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Vectorized calculation
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
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    
    def unpack(params):
        r = params
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        
    def objective(params):
        points = unpack(params)
        ratio = compute_ratio(points)
        return -ratio

    # Bounds for radii to keep points reasonable
    bounds = [(0.1, 2.0)] * n
    
    # Initial parameters
    x0 = np.ones(n)
    
    # Optimize
    # Nelder-Mead is good for small dimensions and non-smooth functions
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    best_points = unpack(best_params)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
