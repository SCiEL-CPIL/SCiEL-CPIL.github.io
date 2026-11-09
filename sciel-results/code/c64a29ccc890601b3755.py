# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _calculate_min_score(polygon):
    """
    Calculate the minimum vertex score for a given polygon.
    Returns 0 if invalid (non-convex, etc.).
    """
    n = polygon.shape[0]
    if n != 20:
        return 0.0
    
    # Normalize
    centroid = np.mean(polygon, axis=0)
    poly_norm = polygon - centroid
    dists_centroid = np.linalg.norm(poly_norm, axis=1)
    mean_dist = np.mean(dists_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    poly_norm = poly_norm / mean_dist
    
    # Check Convexity
    # Cross product of consecutive edges
    # Edge i: P[i+1] - P[i]
    # Edge i+1: P[i+2] - P[i+1]
    # Cross product z-component: (x1*y2 - y1*x2)
    # For CCW convex, cross products should be all positive (or all negative).
    # Let's assume CCW.
    
    for i in range(n):
        p0 = poly_norm[i]
        p1 = poly_norm[(i + 1) % n]
        p2 = poly_norm[(i + 2) % n]
        
        e1 = p1 - p0
        e2 = p2 - p1
        
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        
        # If cross is close to 0, it's not strictly convex (collinear)
        if abs(cross) < 1e-6:
            return 0.0
            
        # Check sign consistency
        # We can check against the first cross product
        if i == 0:
            sign = 1 if cross > 0 else -1
        else:
            if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return 0.0
                
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(poly_norm[i] - poly_norm[j]) < 1e-6:
                return 0.0

    # Calculate Vertex Scores
    min_score = 1.0
    
    for i in range(n):
        vi = poly_norm[i]
        # Distances to other vertices
        diffs = poly_norm - vi
        dists = np.linalg.norm(diffs, axis=1)
        
        # Exclude self-distance (index i)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists = dists[mask]
        
        # Sort distances
        dists_sorted = np.sort(dists)
        
        # Check min distance
        if dists_sorted[0] < 0.05:
            return 0.0
            
        # Find min gap
        gaps = np.diff(dists_sorted)
        k = np.argmin(gaps)
        
        # D
        d_k = dists_sorted[k]
        d_k1 = dists_sorted[k + 1]
        D = (d_k + d_k1) / 2.0
        
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            sel_dists = dists_sorted[0:4]
        elif k == 17: # n-2 = 18, so max k is 18? 
            # dists_sorted has length 19. indices 0..18.
            # gaps has length 18. indices 0..17.
            # So k is in 0..17.
            # If k=17, d_k is index 17, d_k1 is index 18.
            # The rule says: d_15..d_18 if k=17.
            # Indices 15, 16, 17, 18.
            sel_dists = dists_sorted[15:19]
        else:
            # d_{k-1}..d_{k+2}
            # Indices k-1, k, k+1, k+2
            sel_dists = dists_sorted[k-1:k+3]
            
        # Calculate raw
        # max(d/D, D/d)
        ratios = np.maximum(sel_dists / D, D / sel_dists)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial guess: Regular polygon
    # Parameters: radii
    x0 = np.ones(n)
    
    def objective(params):
        radii = np.exp(params) # Ensure positive
        # Optional: constrain radii range to maintain convexity roughly
        # If radii vary too much, convexity might break.
        # Let's clip params to keep radii in [0.5, 2.0]
        # exp(-1) ~ 0.36, exp(1) ~ 2.7
        # Let's use a soft constraint or just rely on Nelder-Mead.
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        score = _calculate_min_score(polygon)
        # We want to maximize score, so minimize -score
        # If score is 0 (invalid), we want a large penalty
        return -score
        
    # Use Nelder-Mead
    # Max iterations to ensure it finishes within time
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = result.x
    best_radii = np.exp(best_params)
    
    x = best_radii * np.cos(angles)
    y = best_radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Final check and normalization
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists_centroid = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists_centroid)
    
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
    else:
        # Fallback to regular
        polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return polygon

# EVOLVE-BLOCK-END
