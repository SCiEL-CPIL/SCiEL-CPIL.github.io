import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def hull_area(points):
    """Compute convex hull area."""
    hull = ConvexHull(points)
    return hull.volume

def score(points):
    """Score = min triangle area / hull area."""
    a_min = min_triangle_area(points)
    a_hull = hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    # Create regular 11-gon for the fixed points
    n_fixed = 11
    angles_fixed = np.linspace(0, 2*np.pi, n_fixed, endpoint=False)
    fixed_pts = np.column_stack([np.cos(angles_fixed), np.sin(angles_fixed)])
    
    # Choose 3 indices to be the "free" points
    free_indices = [0, 4, 8]
    fixed_indices = [i for i in range(n_fixed) if i not in free_indices]
    
    best_score = -np.inf
    best_points = None
    
    # Try multiple initializations for the free points
    for trial in range(5):
        if trial == 0:
            # Start at 11-gon positions
            free_pts_init = fixed_pts[free_indices].copy()
        else:
            # Try different initial positions - spread around
            angle_offset = trial * np.pi / 5
            r = 1.0 + 0.2 * trial
            free_pts_init = np.column_stack([
                r * np.cos(angles_fixed[free_indices] + angle_offset),
                r * np.sin(angles_fixed[free_indices] + angle_offset)
            ])
        
        def objective(x):
            free_pts = x.reshape(3, 2)
            all_pts = np.empty((N, 2))
            all_pts[fixed_indices] = fixed_pts[fixed_indices]
            all_pts[free_indices] = free_pts
            
            try:
                h_area = hull_area(all_pts)
            except Exception:
                return 1e10
            
            if h_area < 1e-6:
                return 1e10
            
            scale = np.sqrt(1.0 / h_area)
            all_pts_scaled = all_pts * scale
            return -min_triangle_area(all_pts_scaled)
        
        result = minimize(objective, free_pts_init.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
        
        free_pts_opt = result.x.reshape(3, 2)
        
        all_pts = np.empty((N, 2))
        all_pts[fixed_indices] = fixed_pts[fixed_indices]
        all_pts[free_indices] = free_pts_opt
        
        h_area = hull_area(all_pts)
        if h_area > 1e-10:
            all_pts = all_pts * np.sqrt(1.0 / h_area)
            s = score(all_pts)
            if s > best_score:
                best_score = s
                best_points = all_pts.copy()
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
