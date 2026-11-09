import numpy as np
from scipy.optimize import minimize

def calculate_vertex_score(vertex, polygon):
    """
    Calculate the score for a single vertex given the full polygon.
    Returns 0.0 if invalid or score is 0.
    """
    n = len(polygon)
    # Distances from vertex to all other vertices
    diffs = polygon - vertex
    # Set distance to self to inf or large number, but we exclude it
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude the vertex itself (index 0 in the diffs array if vertex is polygon[0]?)
    # Actually, diffs[0] is 0. We need the other 19.
    # The problem statement says "19 distances from v to the other vertices".
    # So we take dists[1:] if vertex is at index 0?
    # Better: calculate distances from vertex to ALL vertices, sort, take the 19 non-zero ones?
    # Or just calculate distances to the other 19 explicitly.
    
    # Let's assume 'vertex' is one of the points in 'polygon'.
    # We need the 19 other points.
    # Identify which index 'vertex' is? 
    # In our symmetric setup, we will call this function with specific indices.
    # Let's pass the index as well to be safe, or just assume vertex is not in the list of 'others'.
    # I will modify the function to take the list of OTHER vertices.
    pass

def score_vertex(vertex, others):
    """
    vertex: np.array shape (2,)
    others: np.array shape (19, 2)
    """
    diffs = others - vertex
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    dists = np.sort(dists)
    
    # Check for very small distances
    if np.any(dists < 0.05):
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    # d_{k+1} - d_k for k=0..17
    diffs_arr = dists[1:] - dists[:-1]
    k = np.argmin(diffs_arr)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    
    # Select 4 distances
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    selected_dists = dists[idxs]
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected_dists / D, D / selected_dists)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def compute_total_score(polygon):
    """
    Compute the minimum vertex score over all 20 vertices.
    Assumes polygon is (20, 2).
    """
    n = len(polygon)
    min_score = 1.0
    
    for i in range(n):
        v = polygon[i]
        others = np.delete(polygon, i, axis=0)
        s = score_vertex(v, others)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                break
                
    return min_score

def is_convex(polygon):
    """
    Check if polygon is strictly convex.
    """
    n = len(polygon)
    # Compute cross products of consecutive edges
    # Edge i: P_{i+1} - P_i
    # Edge i+1: P_{i+2} - P_{i+1}
    # Cross product z-component: (x2-x1)(y3-y2) - (y2-y1)(x3-x2)
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i+1) % n]
        p3 = polygon[(i+2) % n]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 0: # Assuming CCW order, cross should be positive
            return False
            
    return True

def generate_polygon(params):
    """
    params: [R1, R2, phi]
    Returns (20, 2) array.
    """
    R1, R2, phi = params
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phi
    radii = np.zeros(n)
    radii[0::2] = R1
    radii[1::2] = R2
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def objective(params):
    """
    Minimize negative score.
    """
    R1, R2, phi = params
    
    # Scale invariance: we can fix R1=1, but optimizer might drift.
    # Let's normalize inside? No, the score is scale invariant, so the landscape is flat along the scale direction.
    # To help the optimizer, we can fix R1=1 and only optimize R2 and phi.
    # Let's change strategy: optimize x = R2/R1 and phi. Set R1=1.
    
    # But the function signature takes 3 params. 
    # Let's just use the 3 params and normalize the polygon before scoring?
    # The task says: "The polygon is normalized: subtract the centroid, divide by the mean distance to it".
    # So the input to the score function in the evaluator is normalized.
    # My `compute_total_score` does NOT normalize.
    # I MUST normalize the polygon before computing the score to match the evaluator's behavior.
    
    polygon = generate_polygon(params)
    
    # Check convexity on the raw polygon (convexity is affine invariant, so raw is fine)
    if not is_convex(polygon):
        return -1.0 # Invalid
    
    # Normalize
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(polygon_norm**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return -1.0
        
    polygon_norm = polygon_norm / mean_dist
    
    score = compute_total_score(polygon_norm)
    
    if score < 0:
        return -1.0
        
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    # Initial guess: Regular polygon
    # R1=1, R2=1, phi=0
    x0 = [1.0, 1.0, 0.0]
    
    # Bounds for R2/R1 and phi
    # R2 can vary. Let's say R1 is fixed to 1 in the search logic, but we have 3 params.
    # To avoid scale issues, let's constrain R1 and R2 to be close to 1.
    # Or better, just optimize R2 and phi with R1 fixed to 1.
    # I will modify the optimization to use 2 variables: [R2, phi] with R1=1.
    
    def obj_2d(params_2d):
        R2, phi = params_2d
        return objective([1.0, R2, phi])
    
    # Grid search to find a good starting point
    best_score = -1.0
    best_params = [1.0, 0.0]
    
    # R2 range [0.5, 1.5], phi range [0, pi/10]
    R2_vals = np.linspace(0.6, 1.4, 10)
    phi_vals = np.linspace(0.0, np.pi/10, 10)
    
    for R2 in R2_vals:
        for phi in phi_vals:
            s = obj_2d([R2, phi])
            if s > best_score:
                best_score = s
                best_params = [R2, phi]
                
    # Local refinement
    if best_score > -0.5: # If we found a valid point
        res = minimize(obj_2d, best_params, method='Nelder-Mead', 
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000})
        
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = -res.fun
            
    # Final check and return
    R2, phi = best_params
    final_params = [1.0, R2, phi]
    polygon = generate_polygon(final_params)
    
    # Ensure it's valid
    if not is_convex(polygon):
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return polygon

# EVOLVE-BLOCK-END
