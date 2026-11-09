import numpy as np
import time
from scipy.optimize import minimize

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

def _evaluate(params):
    """
    params: (20, 2) array of raw coordinates
    Returns: negative of the minimum vertex score (for minimization)
    """
    points = _normalize(params)
    if points is None:
        return 0.0
    if not _is_strictly_convex(points):
        return 0.0
    return -_polygon_score(points)

def _regular_polygon(n=20):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin for 600s limit
    
    n = 20
    best_result = _regular_polygon(n)
    best_score = -_evaluate(best_result)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try multiple random perturbations of the regular polygon
    # Use different scales and optimization methods
    
    # Start from regular polygon
    x0 = best_result.flatten()
    
    # Try L-BFGS-B from regular polygon
    try:
        res = minimize(
            _evaluate, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None:
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_result = res.x.reshape(n, 2)
    except Exception:
        pass
    
    # Try Nelder-Mead from regular polygon
    if time.time() < deadline:
        try:
            res_nm = minimize(
                _evaluate, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_nm.x is not None:
                current_score = -res_nm.fun
                if current_score > best_score:
                    best_score = current_score
                    best_result = res_nm.x.reshape(n, 2)
        except Exception:
            pass
    
    # Phase 2: Multi-restart with random perturbations
    scales = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    for scale in scales:
        if time.time() > deadline:
            break
        
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Random perturbation of regular polygon
            perturb = rng.normal(0, scale, size=(n, 2))
            x_init = x0 + perturb.flatten()
            
            try:
                res = minimize(
                    _evaluate, 
                    x_init, 
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_result = res.x.reshape(n, 2)
                        
            except Exception:
                pass
    
    # Phase 3: Final refinement with Nelder-Mead from best point
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                _evaluate, 
                best_result.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res_nm.x is not None:
                current_score = -res_nm.fun
                if current_score > best_score:
                    best_score = current_score
                    best_result = res_nm.x.reshape(n, 2)
                    
        except Exception:
            pass
    
    # Final validation
    final_points = _normalize(best_result)
    if final_points is None:
        return _regular_polygon(n)
    
    if not _is_strictly_convex(final_points):
        return _regular_polygon(n)
    
    return final_points
