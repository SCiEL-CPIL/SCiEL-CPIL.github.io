import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

def compute_score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Calculate all triangle areas
    # Number of triangles is N*(N-1)*(N-2)/6
    # For N=14, this is 364 triangles. Very fast.
    min_area = np.inf
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, n):
                pk = points[k]
                v2 = pk - pi
                # Area = 0.5 * |cross(v1, v2)|
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def soft_min_score(points, T=0.01):
    """
    Smooth approximation of the min triangle area / hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -1.0
    except Exception:
        return -1.0

    n = len(points)
    areas = []
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
                
    areas = np.array(areas)
    # Soft min: -T * log(sum(exp(-area/T)))
    # To avoid overflow/underflow, subtract max area? No, we want min.
    # min(x) approx -T log(sum(exp(-x/T)))
    # Let's use a stable log-sum-exp for negative values
    neg_areas = -areas / T
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min_val = -T * log_sum
    
    return soft_min_val / hull_area

def normalize_points(points):
    """
    Translate to origin, rotate to align first point to x-axis (if possible), 
    and scale to unit hull area is NOT done here to keep optimization in raw space,
    but we can normalize to improve conditioning.
    Actually, it's better to keep the scale free and just divide by hull area in the score.
    However, fixing the scale helps the optimizer.
    Let's fix the centroid to (0,0) and the first point to (1,0) is hard if we allow rotation.
    Better: Centroid (0,0). Rotate such that the vector from centroid to the point with max x is along x-axis.
    """
    # Translate
    points = points - np.mean(points, axis=0)
    
    # Rotate
    # Find point with max x
    idx = np.argmax(points[:, 0])
    angle = np.arctan2(points[idx, 1], points[idx, 0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    points = points @ R.T
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Stage 1: Optimize the symmetric 7+7 structure
    # Variables: r1, r2, phi
    # r1, r2 > 0
    # phi in [0, 2*pi/7]
    
    def get_sym_points(x):
        r1, r2, phi = x
        if r1 < 0.01 or r2 < 0.01:
            return None
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles2 = angles1 + phi
        
        p1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        p2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        
        points = np.vstack((p1, p2))
        return points

    def obj_sym(x):
        points = get_sym_points(x)
        if points is None:
            return 1.0
        # Use exact score for symmetry, it's cheap enough and avoids soft-min tuning
        return -compute_score(points)

    # Initial guess
    x0 = [0.5, 1.0, np.pi/7]
    
    best_x = x0
    best_score = obj_sym(x0)
    
    # Optimize symmetric parameters
    try:
        res = minimize(obj_sym, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    except Exception:
        pass
        
    # Check time
    if time.time() > deadline:
        points = get_sym_points(best_x)
        return normalize_points(points)
        
    # Stage 2: Full local optimization from the symmetric start
    points_start = get_sym_points(best_x)
    
    # We will optimize the 28 coordinates.
    # To handle invariances, we will re-normalize inside the objective function.
    # This is slightly expensive but robust.
    
    def obj_full(x_flat):
        points = x_flat.reshape(-1, 2)
        # Normalize
        points = normalize_points(points)
        # Calculate soft score or exact score?
        # For local polish, exact min is non-smooth. 
        # Let's use a small T soft-min to help gradients if we used BFGS, 
        # but Nelder-Mead doesn't need gradients.
        # Nelder-Mead is robust to non-smoothness.
        return -compute_score(points)

    x_start = points_start.flatten()
    
    best_x_full = x_start
    best_score_full = obj_full(x_start)
    
    # Try Nelder-Mead with multiple restarts or just one long run?
    # One long run from the good symmetric start is usually best.
    
    try:
        res_full = minimize(obj_full, x_start, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
        if res_full.fun < best_score_full:
            best_x_full = res_full.x
            best_score_full = res_full.fun
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        points = best_x_full.reshape(-1, 2)
        return normalize_points(points)
        
    # Stage 3: Polish with L-BFGS-B using soft-min
    # This can find a tighter minimum if the landscape is smooth enough with small T
    
    T_values = [0.05, 0.01, 0.001]
    current_x = best_x_full
    
    for T in T_values:
        if time.time() > deadline - 5:
            break
            
        def obj_soft(x_flat):
            points = x_flat.reshape(-1, 2)
            points = normalize_points(points)
            s = soft_min_score(points, T)
            return -s

        try:
            res_soft = minimize(obj_soft, current_x, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res_soft.fun < obj_soft(current_x):
                current_x = res_soft.x
        except Exception:
            pass
            
    points_final = current_x.reshape(-1, 2)
    points_final = normalize_points(points_final)
    
    # Final check: ensure it's valid and returns best found
    # We should compare the exact score of the final point vs the best from NM
    score_final = compute_score(points_final)
    score_nm = compute_score(best_x_full.reshape(-1, 2))
    
    if score_final > score_nm:
        return points_final
    else:
        return normalize_points(best_x_full.reshape(-1, 2))
