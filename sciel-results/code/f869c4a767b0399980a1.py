# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _normalize(polygon):
    """
    Normalize the polygon: subtract centroid, divide by mean distance to centroid.
    Returns None if normalization is invalid (mean distance <= 0.01).
    """
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(polygon):
    """
    Check if the polygon is strictly convex and has no repeated points.
    Assumes vertices are in order.
    """
    n = len(polygon)
    if n < 3:
        return False
    
    # Check for repeated points
    # We can check min distance between distinct vertices
    # But for convexity, we can just check cross products.
    # Repeated points would cause zero area or non-strict convexity.
    
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Cross product in 2D: v1.x * v2.y - v1.y * v2.x
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Nonzero turn
        if np.abs(cross) < 1e-10:
            return False
        
        signs.append(np.sign(cross))
        
    # All turns must have the same sign
    if not (all(s == signs[0] for s in signs)):
        return False
        
    return True

def _score_polygon_vectorized(polygon):
    """
    Calculate the score for a polygon.
    Returns the minimum vertex score.
    Returns -inf if invalid.
    """
    normalized = _normalize(polygon)
    if normalized is None:
        return -np.inf
        
    if not _is_strictly_convex(normalized):
        return -np.inf
        
    n = len(normalized)
    # Compute pairwise distances
    # dist[i, j] is distance between vertex i and j
    # We need distances from each vertex to all others, sorted.
    
    # Vectorized distance calculation
    # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # dists[i] is the vector of distances from vertex i to all vertices j.
    # dists[i, i] is 0.
    
    # We need to sort the 19 distances from each vertex to the OTHER vertices.
    # So we set diagonal to inf, sort, and take first 19.
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)[:, :19]
    
    # sorted_dists[i] = [d_0, d_1, ..., d_18] for vertex i
    
    # Check d_0 < 0.05
    min_dists = sorted_dists[:, 0]
    invalid_mask = min_dists < 0.05
    
    # Calculate gaps: d_{k+1} - d_k
    # We need to find k that minimizes the gap.
    # Gaps are defined for k=0..17 (between d_k and d_{k+1})
    gaps = np.diff(sorted_dists, axis=1) # shape (n, 18)
    
    # Find index of minimum gap
    k_indices = np.argmin(gaps, axis=1)
    
    # Determine the 4 distances to use for each vertex
    # If k == 0: d_0, d_1, d_2, d_3
    # If k == 17: d_15, d_16, d_17, d_18
    # Else: d_{k-1}, d_k, d_{k+1}, d_{k+2}
    
    # We will build an array of indices to select from sorted_dists
    # Shape (n, 4)
    
    # Initialize with zeros
    indices = np.zeros((n, 4), dtype=int)
    
    # Case 1: k == 0
    mask_0 = (k_indices == 0)
    indices[mask_0, 0] = 0
    indices[mask_0, 1] = 1
    indices[mask_0, 2] = 2
    indices[mask_0, 3] = 3
    
    # Case 2: k == 17
    mask_17 = (k_indices == 17)
    indices[mask_17, 0] = 15
    indices[mask_17, 1] = 16
    indices[mask_17, 2] = 17
    indices[mask_17, 3] = 18
    
    # Case 3: Else
    mask_else = ~(mask_0 | mask_17)
    k_else = k_indices[mask_else]
    # d_{k-1}
    indices[mask_else, 0] = k_else - 1
    # d_k
    indices[mask_else, 1] = k_else
    # d_{k+1}
    indices[mask_else, 2] = k_else + 1
    # d_{k+2}
    indices[mask_else, 3] = k_else + 2
    
    # Use advanced indexing to get the 4 distances for each vertex
    # sorted_dists[i, indices[i, j]]
    # We can do: sorted_dists[np.arange(n)[:, None], indices]
    row_indices = np.arange(n)[:, None]
    selected_dists = sorted_dists[row_indices, indices] # shape (n, 4)
    
    # Calculate D = (d_k + d_{k+1}) / 2
    # We need d_k and d_{k+1} specifically.
    # d_k is at index k_indices
    # d_{k+1} is at index k_indices + 1
    d_k = sorted_dists[np.arange(n), k_indices]
    d_k_plus_1 = sorted_dists[np.arange(n), k_indices + 1]
    D = (d_k + d_k_plus_1) / 2.0
    
    # Check D < 0.1
    invalid_mask |= (D < 0.1)
    
    # Calculate raw score
    # raw = mean over the 4 d's of max(d/D, D/d)
    # Let's compute ratio = d / D
    # ratio = max(ratio, 1/ratio)
    # Note: D might be 0? No, D >= 0.1.
    # Also d's are >= 0.05.
    
    ratios = selected_dists / D[:, None]
    # max(d/D, D/d) is equivalent to max(ratio, 1/ratio)
    # Since d, D > 0, this is symmetric.
    # If ratio < 1, 1/ratio > 1. If ratio > 1, ratio > 1.
    # So max(ratio, 1/ratio) = ratio if ratio >= 1 else 1/ratio.
    # This is equivalent to: np.where(ratio >= 1, ratio, 1/ratio)
    
    # However, we must be careful if D is 0, but we checked D >= 0.1.
    # Also selected_dists are distances, so > 0.
    
    raw_components = np.where(ratios >= 1.0, ratios, 1.0 / ratios)
    raw = np.mean(raw_components, axis=1)
    
    # vertex score = min(1, 1/raw)
    scores = np.minimum(1.0, 1.0 / raw)
    
    # Mark invalid vertices as -inf
    scores[invalid_mask] = -np.inf
    
    # Total score is minimum vertex score
    return np.min(scores)

def _objective(params, polygon_shape):
    """
    Objective function for optimizer.
    params: flat array of 2*n coordinates.
    Returns negative of the score (to minimize).
    """
    n = polygon_shape
    polygon = params.reshape((n, 2))
    score = _score_polygon_vectorized(polygon)
    if np.isneginf(score) or np.isnan(score):
        return 1e10 # Large penalty
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, keep margin
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_polygon = np.column_stack([np.cos(angles), np.sin(angles)])
    init_params = init_polygon.flatten()
    
    best_params = init_params.copy()
    best_score = _score_polygon_vectorized(best_params.reshape(n, 2))
    
    if np.isneginf(best_score):
        # Should not happen for regular polygon, but just in case
        return best_params.reshape(n, 2)
        
    # Optimization strategy
    # Use Nelder-Mead.
    # We will do multiple restarts with different noise.
    
    # Define a set of noise scales
    noise_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    # We will iterate through noise scales and perform optimization
    # If time runs out, we return the best found so far.
    
    # To save time, we can limit the number of function evaluations per restart
    # and the number of restarts.
    
    # Let's try a few restarts.
    # Nelder-Mead options:
    # maxiter, xatol, fatol
    
    # We'll use a relatively tight tolerance for polishing, but allow some iterations.
    
    rng = np.random.default_rng(42)
    
    for noise_scale in noise_scales:
        if time.time() > deadline - 60.0: # Stop if less than 60s left
            break
            
        # Generate initial parameters
        if noise_scale == 0.0:
            current_params = init_params.copy()
        else:
            current_params = init_params + rng.normal(0, noise_scale, size=init_params.shape)
        
        # Check if current params are valid
        current_score = _score_polygon_vectorized(current_params.reshape(n, 2))
        if not np.isneginf(current_score) and current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            
        # Optimize
        # We want to maximize score, so minimize -score
        try:
            res = minimize(_objective, current_params, args=(n,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < -best_score:
                # Check the result
                res_score = _score_polygon_vectorized(res.x.reshape(n, 2))
                if not np.isneginf(res_score) and res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass # Ignore optimization errors and continue
            
    # Final check
    final_polygon = best_params.reshape(n, 2)
    final_score = _score_polygon_vectorized(final_polygon)
    
    # If the best score is invalid, fall back to regular polygon
    if np.isneginf(final_score):
         final_polygon = init_params.reshape(n, 2)
         
    return final_polygon

# EVOLVE-BLOCK-END
