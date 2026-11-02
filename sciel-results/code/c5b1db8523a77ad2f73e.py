import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    # Ensure points are in a valid convex position? 
    # The problem says "Place 14 points in a planar convex region". 
    # It does not require the points themselves to be in convex position (i.e., on the hull).
    # However, if a point is inside the hull, it might form smaller triangles.
    # The score is normalized by the convex hull area of the *point set*.
    
    # Calculate Convex Hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, .volume is area
    except Exception:
        # If all points are collinear or identical, area is 0
        return 0.0

    if hull_area == 0:
        return 0.0

    # Calculate all triangle areas
    # N=14, C(14,3) = 364 triangles. This is fast.
    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |det(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def construct():
    # Initial guess: Jittered 3x5 grid minus one point
    x_vals = np.linspace(0, 4, 5)
    y_vals = np.linspace(0, 2, 3)
    grid_points = np.array([[x, y] for x in x_vals for y in y_vals])
    points = grid_points[1:].copy()
    
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    start_time = time.time()
    timeout = 100.0 # Leave 20s buffer for polish
    
    current_points = points.copy()
    current_score = compute_score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 0.05
    T_min = 1e-4
    alpha = 0.995
    step_size = 0.3
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout:
        # Generate a neighbor
        new_points = current_points.copy()
        idx = np.random.randint(0, N)
        new_points[idx] += np.random.normal(0, step_size, 2)
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            if np.random.rand() < np.exp((new_score - current_score) / T):
                current_points = new_points
                current_score = new_score
            no_improve_count += 1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            current_points = best_points.copy()
            current_score = best_score
            T = 0.01
            no_improve_count = 0
            step_size = 0.1
            
    # Final Polish with Nelder-Mead
    if time.time() - start_time < timeout - 5:
        def neg_score(p):
            pts = p.reshape(N, 2)
            return -compute_score(pts)
            
        try:
            res = minimize(neg_score, best_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            if -res.fun > best_score:
                best_points = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass
            
    return best_points
