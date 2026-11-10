import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def calculate_score(polygon):
    """
    Calculate the minimum vertex score for a given polygon.
    Returns 0 if invalid (non-convex, etc).
    """
    n = len(polygon)
    if n != 20:
        return 0.0
        
    # Normalize
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    normalized = centered / mean_dist
    
    # Check Convexity
    # Calculate cross products of consecutive edges
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross_products = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    # All cross products must be non-zero and same sign
    if np.any(cross_products == 0):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
        
    # Calculate distances from each vertex to all others
    # dist_matrix[i, j] is distance from vertex i to vertex j
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    min_score = 1.0
    
    for i in range(n):
        # Distances to other vertices (exclude self, which is 0)
        d = dist_matrix[i].copy()
        d[i] = np.inf
        # Sort the 19 distances
        d_sorted = np.sort(d)
        
        # Check for very small distances (degenerate)
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
            
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = d_sorted[1:] - d_sorted[:-1]
        # We need the first index k such that d_{k+1} - d_k is minimal.
        # Note: d_sorted has 19 elements (indices 0..18).
        # diffs has 18 elements (indices 0..17).
        # The problem statement says: "k be the first index minimizing d_{k+1} - d_k".
        # Indices of d are 0..18.
        # So we look at diffs[0] (d1-d0), diffs[1] (d2-d1), ..., diffs[17] (d18-d17).
        # If the minimum is at index m in diffs, then k = m.
        # Wait, "first index minimizing".
        min_diff_val = np.min(diffs)
        # Find first occurrence
        k = np.where(diffs == min_diff_val)[0][0]
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            min_score = 0.0
            break
            
        # Select 4 distances
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw)
        
        if vertex_score < min_score:
            min_score = vertex_score
            
    return min_score

def objective(params):
    """
    Objective function for minimization.
    We want to MAXIMIZE the score, so we minimize -score.
    params: array of 20 radii.
    """
    r = params
    # Ensure positive radii
    r = np.maximum(r, 0.1)
    
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    score = calculate_score(polygon)
    
    # If score is 0, return a high penalty to steer away
    if score == 0.0:
        return 100.0
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    n = 20
    best_score = -np.inf
    best_params = np.ones(n) # Regular polygon
    
    # Initial best is regular polygon
    angles_init = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    poly_init = np.stack([np.cos(angles_init), np.sin(angles_init)], axis=1)
    best_score = calculate_score(poly_init)
    
    # Differential Evolution bounds
    # Radii should be close to 1. Let's say [0.5, 1.5]
    bounds = [(0.5, 1.5)] * n
    
    # Run DE
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=1e-5, 
            mutation=(0.3, 1.0), 
            recombination=0.9,
            seed=42,
            polish=False
        )
        if de_result.fun < -best_score:
            best_params = de_result.x
            best_score = -de_result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return _params_to_polygon(best_params)
        
    # Polish with Nelder-Mead from best DE result and a few random starts
    local_starts = [best_params]
    # Add a few random perturbations of the regular polygon
    for i in range(5):
        if time.time() > deadline: break
        r_pert = 1.0 + 0.2 * np.random.normal(0, 1, n)
        r_pert = np.clip(r_pert, 0.5, 1.5)
        local_starts.append(r_pert)
        
    for start_p in local_starts:
        if time.time() > deadline: break
        try:
            lm_result = minimize(
                objective, 
                start_p, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if lm_result.fun < -best_score:
                best_params = lm_result.x
                best_score = -lm_result.fun
        except Exception:
            pass
            
    return _params_to_polygon(best_params)

def _params_to_polygon(params):
    r = np.maximum(params, 0.1)
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.stack([x, y], axis=1)

# EVOLVE-BLOCK-END
