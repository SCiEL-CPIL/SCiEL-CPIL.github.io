import numpy as np
import time
from scipy.optimize import minimize

def calculate_scores(points):
    """
    Calculate the vertex scores for a given polygon.
    Returns an array of 20 scores.
    """
    n = len(points)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        # Distances from i to all others, exclude self
        d = np.delete(dists[i], i)
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check for repeated points (dist < 0.05)
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
            continue
            
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(diffs)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Select 4 distances
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17: # n-3? 19-1=18 indices. 17 is second to last gap?
            # Indices 0..18. Gaps 0..17.
            # If k=17, gap is d_18 - d_17.
            # Rule: d_15..d_18
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def min_vertex_score(radii):
    """
    radii: array of 20 radii.
    Returns the minimum vertex score.
    """
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    # Normalize: subtract centroid, divide by mean dist to centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    points = points / mean_dist
    
    # Check convexity
    # Compute cross products of consecutive edges
    # Edge i: p_{i+1} - p_i
    # Cross product z-component: (p_{i+1}-p_i) x (p_{i+2}-p_{i+1})
    # (x_{i+1}-x_i)*(y_{i+2}-y_{i+1}) - (y_{i+1}-y_i)*(x_{i+2}-x_{i+1})
    p = points
    p_next = np.roll(p, -1, axis=0)
    p_next2 = np.roll(p, -2, axis=0)
    
    e1 = p_next - p
    e2 = p_next2 - p_next
    
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    
    # Strictly convex: all cross products non-zero and same sign
    if np.any(cross == 0):
        return 0.0
    if np.any(cross > 0) and np.any(cross < 0):
        return 0.0
        
    # Also check no repeated points
    if np.min(np.linalg.norm(p_next - p, axis=1)) < 1e-9:
         return 0.0
         
    scores = calculate_scores(points)
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0 # 50s safety margin for 630s limit
    
    n = 20
    # Initial guess: regular polygon
    x0 = np.ones(n)
    
    # We want to maximize min_vertex_score
    # scipy.minimize minimizes, so we negate
    def objective(r):
        # Ensure radii are positive to maintain convexity/validity
        r = np.maximum(r, 0.1)
        return -min_vertex_score(r)
        
    best_result = None
    best_score = -1.0
    
    # Use Nelder-Mead as it handles non-smooth min well
    # We will try multiple restarts with different initial patterns if time permits
    
    try:
        # Restart 1: Regular
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
        if res.fun < best_score:
            best_score = -res.fun
            best_result = res.x
            
        # Restart 2: Wavy 4 bumps (symmetry breaking)
        if time.time() < deadline - 60:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            x1 = 1.0 + 0.1 * np.cos(4*angles)
            x1 = np.maximum(x1, 0.1)
            res2 = minimize(objective, x1, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res2.fun < best_score:
                best_score = -res2.fun
                best_result = res2.x
                
        # Restart 3: Wavy 5 bumps (symmetry breaking)
        if time.time() < deadline - 60:
            x3 = 1.0 + 0.1 * np.cos(5*angles)
            x3 = np.maximum(x3, 0.1)
            res3 = minimize(objective, x3, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res3.fun < best_score:
                best_score = -res3.fun
                best_result = res3.x

        # Restart 4: Wavy 2 bumps
        if time.time() < deadline - 30:
            x4 = 1.0 + 0.1 * np.cos(2*angles)
            x4 = np.maximum(x4, 0.1)
            res4 = minimize(objective, x4, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res4.fun < best_score:
                best_score = -res4.fun
                best_result = res4.x

        # Restart 5: Random perturbation
        if time.time() < deadline - 30:
            np.random.seed(42)
            x5 = 1.0 + 0.05 * np.random.randn(n)
            x5 = np.maximum(x5, 0.1)
            res5 = minimize(objective, x5, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res5.fun < best_score:
                best_score = -res5.fun
                best_result = res5.x

    except Exception:
        # Fallback to regular
        pass
        
    if best_result is None:
        best_result = x0
        
    # Construct final polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = best_result * np.cos(angles)
    y = best_result * np.sin(angles)
    polygon = np.column_stack((x, y))
    
    # Final Normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
        
    return polygon
