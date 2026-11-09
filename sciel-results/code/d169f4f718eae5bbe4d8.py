import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize, differential_evolution

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized min triangle area
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def fast_score(points):
    """
    Fast score calculation for inner optimization loops.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def score_circle(angles):
    """
    Calculate score for points on the unit circle defined by angles.
    """
    # Ensure angles are sorted for hull calculation efficiency, 
    # though ConvexHull handles unsorted. 
    # For min triangle area, order doesn't matter.
    # But for hull area, if we compute analytically, sorting helps.
    # We will use ConvexHull for robustness.
    
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    return fast_score(points)

def local_refine_powell(points, deadline, max_iter=500):
    """
    Local refinement using Powell's method.
    """
    current_points = points.copy()
    current_score = get_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -fast_score(pts)
    
    try:
        res = minimize(objective, current_points.flatten(), method='Powell', 
                      options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-10})
        
        if time.time() < deadline - 2.0:
            if -res.fun > best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
    except Exception:
        pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: Regular N-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    best_points = initial_points.copy()
    best_score = get_score(best_points)
    
    # 1. Optimize angles on the circle (1D circular packing)
    # This enforces points on the hull, maximizing hull area for the spread.
    def circle_objective(angles_flat):
        # Angles might wrap around. We can normalize them or just use as is.
        # Cos/Sin handles any angle.
        return -score_circle(angles_flat)
    
    initial_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    try:
        # Use Powell's method on the angles
        res_ang = minimize(circle_objective, initial_angles, method='Powell',
                           options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
        
        if time.time() < deadline - 30.0:
            candidate_points = np.column_stack((np.cos(res_ang.x), np.sin(res_ang.x)))
            candidate_score = get_score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
    except Exception:
        pass
    
    # 2. Local Refinement in 2D from the best circular solution
    if time.time() < deadline - 20.0:
        best_points, best_score = local_refine_powell(best_points, deadline, max_iter=300)
        
    # 3. Differential Evolution in 2D if time permits
    if time.time() < deadline - 10.0:
        def objective(flat_points):
            pts = flat_points.reshape(N, 2)
            return -fast_score(pts)
            
        bounds = [(-2, 2)] * (2 * N)
        
        try:
            # Use a smaller population to fit in time
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=42,
                maxiter=10,
                popsize=8,
                tol=0.01,
                polish=False
            )
            
            if time.time() < deadline - 5.0:
                if -res_de.fun > best_score:
                    best_score = -res_de.fun
                    best_points = res_de.x.reshape(N, 2)
                    
                    # Final polish
                    if time.time() < deadline - 2.0:
                        best_points, best_score = local_refine_powell(best_points, deadline, max_iter=100)
                        
        except Exception:
            pass
            
    return best_points
