import numpy as np
import time
import scipy.optimize as opt
import scipy.spatial as spat

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute minimum triangle area
    # N=13, C(13,3) = 286 triangles
    pts = points
    n = len(pts)
    min_area = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle formed by pts[i], pts[j], pts[k]
                # 0.5 * | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) |
                area = 0.5 * abs(
                    pts[i,0]*(pts[j,1] - pts[k,1]) + 
                    pts[j,0]*(pts[k,1] - pts[i,1]) + 
                    pts[k,0]*(pts[i,1] - pts[j,1])
                )
                if area < min_area:
                    min_area = area
                    if min_area < 1e-9:
                        return 0.0 # Degenerate

    return min_area / hull_area

def construct():
    start_time = time.time()
    
    # Symmetric Initialization
    # 1 center, 6 inner, 6 outer
    # Inner radius 0.4, Outer radius 1.0
    # Angles: Inner at 0, 60, ..., Outer at 30, 90, ... (staggered)
    
    pts = np.zeros((N, 2))
    # Center
    pts[0] = [0.0, 0.0]
    
    # Inner Hexagon
    r1 = 0.4
    for i in range(6):
        angle = i * np.pi / 3
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    # Outer Hexagon (staggered by 30 degrees)
    r2 = 1.0
    for i in range(6):
        angle = (i * np.pi / 3) + np.pi / 6
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]

    def objective(flat_pts):
        p = flat_pts.reshape(N, 2)
        # Check time limit
        if time.time() - start_time > 120:
            return 1e10 # Penalize heavily to stop search
        score = calculate_score(p)
        if score == 0.0:
            return 1e10
        return -score

    # Use Powell's method as suggested by the knowledge base
    # Powell is good for non-smooth functions and doesn't require gradients
    try:
        result = opt.minimize(
            objective, 
            pts.flatten(), 
            method='Powell',
            options={'maxiter': 1000, 'maxfev': 5000}
        )
    except Exception:
        # Fallback to original points if optimization fails
        return pts

    optimized_pts = result.x.reshape(N, 2)
    
    # Verify the result is valid and better than start
    score_opt = calculate_score(optimized_pts)
    score_init = calculate_score(pts)
    
    if score_opt > score_init:
        return optimized_pts
    else:
        return pts
