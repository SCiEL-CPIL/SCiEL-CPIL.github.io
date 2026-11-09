# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _compute_min_vertex_score(angles):
    """
    Computes the minimum vertex score for a cyclic polygon defined by angles.
    angles: array of 20 angles in [0, 2*pi)
    """
    n = 20
    # Generate points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.column_stack((x, y))
    
    # Normalization: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    dists_to_centroid = np.linalg.norm(pts_centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist < 0.01:
        return 0.0
        
    pts_normalized = pts_centered / mean_dist
    
    # Check for repeated points
    # For efficiency, we can skip this if we trust the parameterization,
    # but it's good to be safe.
    # min distance between any pair
    # This is O(N^2).
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts_normalized[i] - pts_normalized[j])
            if d < min_pair_dist:
                min_pair_dist = d
            if d < 1e-8:
                return 0.0
                
    # Calculate vertex scores
    min_score = 1.0
    
    # Precompute pairwise distances
    diff = pts_normalized[:, np.newaxis, :] - pts_normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    for i in range(n):
        d = dist_matrix[i].copy()
        d[i] = np.inf
        
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            return 0.0
            
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            selected_d = d_sorted[0:4]
        elif k == 17:
            selected_d = d_sorted[15:19]
        else:
            selected_d = d_sorted[k-1:k+3]
            
        ratios = np.maximum(selected_d / D, D / selected_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw)
        
        if vertex_score < min_score:
            min_score = vertex_score
            
    return min_score

def equidistant_points_polygon20():
    n = 20
    start_time = time.time()
    deadline = start_time + 550 # Safety margin
    
    def angles_from_params(x):
        w = np.exp(x - np.max(x))
        sum_w = np.sum(w)
        gaps = w / sum_w * 2 * np.pi
        angles = np.cumsum(np.concatenate(([0.0], gaps)))[:-1]
        return angles
        
    def objective(x):
        angles = angles_from_params(x)
        score = _compute_min_vertex_score(angles)
        return -score
        
    bounds = [(-10.0, 10.0)] * (n - 1)
    x0 = np.zeros(n - 1)
    
    best_x = x0
    best_score = _compute_min_vertex_score(angles_from_params(x0))
    
    def callback(xk, convergence):
        nonlocal best_x, best_score
        current_score = -objective(xk)
        if current_score > best_score:
            best_score = current_score
            best_x = xk.copy()
        if time.time() > deadline:
            raise Exception("Time limit reached")
            
    try:
        res = differential_evolution(objective, bounds, maxiter=10000, popsize=20, 
                                      init='latinhypercube', seed=42, 
                                      callback=callback, polish=False)
        if -res.fun > best_score:
            best_score = -res.fun
            best_x = res.x
    except Exception as e:
        if "Time limit reached" not in str(e):
            pass # Ignore other errors, use best found
            
    angles = angles_from_params(best_x)
    
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0.01:
        polygon = polygon / mean_dist
        
    return polygon

# EVOLVE-BLOCK-END
