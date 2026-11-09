import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _compute_min_triangle_area(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    """
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _compute_hull_area(pts):
    """
    Compute the area of the convex hull of the points.
    """
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-9

def score(pts):
    """
    Calculate the score: min triangle area divided by hull area.
    """
    min_area = _compute_min_triangle_area(pts)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    return min_area / hull_area

def objective(pts_flat):
    """
    Objective function for minimization (negative of score).
    """
    pts = pts_flat.reshape(N, 2)
    return -score(pts)

def construct():
    best_score = 0.0
    best_points = None
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # 10s buffer for 120s limit
    
    n_restarts = 30
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization strategies
        if r == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        elif r == 1:
            # Regular polygon rotated
            angles = np.linspace(0.1, 2 * np.pi + 0.1, N, endpoint=False)
            init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Random points in unit square
            rng = np.random.default_rng(r)
            init_pts = rng.random((N, 2)) * 2 - 1 # [-1, 1]
            
        # Normalize initial points to roughly unit area to help optimizer
        try:
            hull = ConvexHull(init_pts)
            area = hull.volume
            if area > 1e-6:
                init_pts = init_pts / np.sqrt(area)
        except Exception:
            pass
        
        # Run Nelder-Mead optimization
        try:
            res = minimize(objective, init_pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-9})
            
            if res.success:
                final_pts = res.x.reshape(N, 2)
                s = score(final_pts)
                if s > best_score:
                    best_score = s
                    best_points = final_pts.copy()
        except Exception:
            continue

    if best_points is None:
        # Fallback to regular polygon if optimization fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
