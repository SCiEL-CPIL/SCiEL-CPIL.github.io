# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _vertex_score(distances, D):
    # distances: sorted array of 19 distances
    # D: gap center
    # Returns score for this vertex
    if np.any(distances < 0.05):
        return 0.0
    if D < 0.1:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    diffs = np.diff(distances)
    k = np.argmin(diffs)
    
    # Select 4 distances
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17:
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    d_sel = distances[idxs]
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    # polygon: (20, 2)
    # Normalize
    centroid = np.mean(polygon, axis=0)
    poly = polygon - centroid
    radii = np.linalg.norm(poly, axis=1)
    mean_r = np.mean(radii)
    if mean_r <= 0.01:
        return 0.0
    poly = poly / mean_r
    
    n = len(poly)
    min_score = 1.0
    for i in range(n):
        v = poly[i]
        # Distances to other vertices
        diffs = poly - v
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self (index i)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists = dists[mask]
        
        # Sort
        dists = np.sort(dists)
        
        # Calculate score for this vertex
        # We need D first, which depends on k, which depends on diffs
        diffs = np.diff(dists)
        k = np.argmin(diffs)
        D = (dists[k] + dists[k+1]) / 2.0
        
        s = _vertex_score(dists, D)
        if s < min_score:
            min_score = s
            
    return min_score

def _objective(theta):
    # theta: array of 20 angles
    # Ensure convexity
    # Sort angles? No, the order matters for the polygon.
    # The input theta is the cyclic order.
    # We assume theta is sorted.
    # Check convexity: cross products of consecutive edges
    x = np.cos(theta)
    y = np.sin(theta)
    poly = np.stack([x, y], axis=1)
    
    # Check convexity
    # v1 = p1 - p0, v2 = p2 - p1, etc.
    # Cross product z-component should be all positive (or all negative)
    # For CCW, cross > 0
    edges = np.roll(poly, -1, axis=0) - poly
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(cross <= 0):
        # Not strictly convex or self-intersecting
        return 1000.0
    
    score = _min_vertex_score(poly)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial regular polygon
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Optimization
    # Use Nelder-Mead
    start_time = time.time()
    deadline = start_time + 600
    
    best_score = -_objective(theta0)
    best_theta = theta0.copy()
    
    # Run Powell optimization
    try:
        res = opt.minimize(_objective, theta0, method='Powell', options={'maxiter': 1000, 'maxfev': 5000})
        if res.fun < -best_score:
            best_score = -res.fun
            best_theta = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        pass
    else:
        # Try Nelder-Mead for refinement or different start
        try:
            res2 = opt.minimize(_objective, best_theta, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_theta = res2.x
        except Exception:
            pass
            
    # Final check
    x = np.cos(best_theta)
    y = np.sin(best_theta)
    polygon = np.stack([x, y], axis=1)
    
    return polygon

# EVOLVE-BLOCK-END
