import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 13

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        # Check for degenerate points
        if len(points) < 3:
            return 0.0
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute Min Triangle Area
        # Number of triangles C(13, 3) = 286
        # Vectorized calculation
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def neg_score_smooth(points_flat, k=5):
    """
    Smoothed objective: use k-th smallest triangle area instead of minimum.
    This creates a smoother landscape for optimization.
    """
    try:
        points = points_flat.reshape(N, 2)
        
        # Compute Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Compute triangle areas
        idx = np.array(list(itertools.combinations(range(N), 3)))
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Use k-th smallest area as surrogate
        k = min(k, len(areas))
        kth_area = np.partition(areas, k-1)[k-1]
        
        return -kth_area / hull_area
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Precompute triangle index combinations for speed
    tri_idx = np.array(list(itertools.combinations(range(N), 3)))
    
    # Define bounds for differential evolution
    # Points in [-2, 2] x [-2, 2] should cover the relevant space
    bounds = [(-2.0, 2.0)] * (N * 2)
    
    # Strategy 1: Differential Evolution (global search)
    # Use smoothed objective for smoother landscape
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                neg_score_smooth,
                bounds=bounds,
                args=(5,),  # k=5 for smoothing
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,  # We'll polish separately
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.success and de_result.fun < 0:
                de_points = de_result.x.reshape(N, 2)
                de_score = score(de_points)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points
    except Exception:
        pass
    
    # Strategy 2: L-BFGS-B polish from multiple starting points
    # Generate diverse starting points
    starting_points = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_13 = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points.append(init_13)
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    init_12c = np.vstack([pts_12, np.array([[0.0, 0.0]])])
    starting_points.append(init_12c)
    
    # 3. Two rings: 7 outer, 6 inner
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_inner = 0.5
    pts_7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
    pts_6 = np.column_stack((r_inner * np.cos(angles_6), r_inner * np.sin(angles_6)))
    init_2ring = np.vstack([pts_7, pts_6])
    starting_points.append(init_2ring)
    
    # 4. Noisy versions
    for _ in range(4):
        noise = rng.normal(0, 0.2, (N, 2))
        starting_points.append(init_13 + noise)
    
    # 5. If DE found a good point, add it and noisy versions
    if best_points is not None:
        starting_points.append(best_points)
        for _ in range(2):
            noise = rng.normal(0, 0.1, (N, 2))
            starting_points.append(best_points + noise)
    
    # L-BFGS-B optimization from each starting point
    for i, start in enumerate(starting_points):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                neg_score, 
                start.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            cur_score = -res.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass
        
        # Also try with smoothed objective, then polish
        if time.time() > deadline - 10:
            break
        try:
            res_smooth = minimize(
                neg_score_smooth, 
                start.flatten(), 
                args=(3,),  # Less smoothing for final stage
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            # Now polish with true objective
            res_final = minimize(
                neg_score, 
                res_smooth.x, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            cur_score = -res_final.fun
            if cur_score > best_score:
                best_score = cur_score
                best_points = res_final.x.reshape(N, 2)
                
        except Exception:
            pass

    # If no optimization worked or failed, return the regular polygon
    if best_points is None or best_score <= 0:
        best_points = init_13
    else:
        # Verify the best point is valid
        final_score = score(best_points)
        if final_score <= 0:
            best_points = init_13

    return best_points
