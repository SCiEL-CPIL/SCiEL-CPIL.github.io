import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _normalize(points):
    """Normalize: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return None
    return centered / mean_dist

def _is_strictly_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return False
    # Check all turns are same sign and nonzero
    turns = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        turns.append(cross)
    turns = np.array(turns)
    # All must be same sign and nonzero
    if np.any(np.abs(turns) < 1e-10):
        return False
    if np.any(turns > 0) and np.any(turns < 0):
        return False
    return True

def _vertex_score(v_idx, points):
    """Calculate score for a single vertex."""
    # Distances to all other points
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf  # Exclude self
    
    # Sort 19 distances
    sorted_dists = np.sort(dists)
    
    # Check for repeated points (dist < 0.05)
    if sorted_dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
        
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_score(points):
    """Calculate the minimum vertex score over all vertices."""
    n = len(points)
    scores = [_vertex_score(i, points) for i in range(n)]
    return min(scores)

def _get_points_from_coeffs(coeffs, n=20):
    """
    Generate points from radial modulation coefficients.
    r(theta) = 1 + sum(c_k * cos(k * theta))
    theta_i = 2 * pi * i / n
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = np.ones(n)
    for k, c in enumerate(coeffs, start=1):
        radii += c * np.cos(k * angles)
    
    # Ensure radii are positive (convexity safeguard)
    # If any radius <= 0, this configuration is invalid
    if np.any(radii <= 0.1): # Threshold to avoid near-singularity
        return None
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _objective(coeffs):
    """
    Objective function for minimization.
    Returns negative of the polygon score.
    """
    points = _get_points_from_coeffs(coeffs)
    if points is None:
        return 1e6
        
    normalized_points = _normalize(points)
    if normalized_points is None:
        return 1e6
        
    if not _is_strictly_convex(normalized_points):
        return 1e6
        
    score = _polygon_score(normalized_points)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    n = 20
    # Start with regular polygon (all coeffs 0)
    best_coeffs = np.zeros(8)
    best_points = _get_points_from_coeffs(best_coeffs)
    best_score = -_objective(best_coeffs)
    
    # Define bounds for coefficients
    # Small coefficients ensure convexity. 
    # Let's try range [-0.5, 0.5] for low order, smaller for high order.
    bounds = []
    for k in range(1, 9):
        # Decay bound to ensure stability
        bound_val = 0.5 / (k ** 0.5)
        bounds.append((-bound_val, bound_val))
        
    # Phase 1: Differential Evolution
    # Use a moderate population size and maxiter to explore the space
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False, # We will polish separately
            callback=None
        )
        
        if de_result.x is not None:
            current_score = -de_result.fun
            if current_score > best_score:
                best_score = current_score
                best_coeffs = de_result.x
                best_points = _get_points_from_coeffs(best_coeffs)
                
    except Exception:
        pass
        
    # Phase 2: Local Refinement with L-BFGS-B
    # Try to refine the best solution found so far
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _objective, 
                best_coeffs, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_coeffs = res.x
                    best_points = _get_points_from_coeffs(best_coeffs)
        except Exception:
            pass

    # Phase 3: Multi-restart Local Optimization
    # If we have time, try a few random restarts of local optimization
    # from points near the best solution or random valid points
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            
            # Perturb the best coefficients
            perturb_scale = 0.1
            new_coeffs = best_coeffs + rng.normal(0, perturb_scale, size=8)
            # Clip to bounds
            for i, (lo, hi) in enumerate(bounds):
                new_coeffs[i] = np.clip(new_coeffs[i], lo, hi)
                
            try:
                res = minimize(
                    _objective, 
                    new_coeffs, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_coeffs = res.x
                        best_points = _get_points_from_coeffs(best_coeffs)
            except Exception:
                pass

    # Final validation and return
    if best_points is None:
        return _get_points_from_coeffs(np.zeros(8))
        
    final_points = _normalize(best_points)
    if final_points is None:
        return _get_points_from_coeffs(np.zeros(8))
        
    if not _is_strictly_convex(final_points):
        return _get_points_from_coeffs(np.zeros(8))
        
    return final_points
