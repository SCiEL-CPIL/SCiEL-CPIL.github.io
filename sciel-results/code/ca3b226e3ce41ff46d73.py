# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex in [0, 1].
    """
    d = np.sort(distances)
    
    if np.any(d < 0.05):
        return 0.0
        
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score


def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                return False
                
    # Check turns
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(np.abs(cross) < 1e-8):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    return True


def _polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score after normalization.
    """
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_norm / mean_dist
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    min_score = 1.0
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = _vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score


def _regular_polygon_points():
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


def _params_to_points(params):
    """
    params: array of length 40 - first 20 are radii, next 20 are angular offsets (in radians)
    Returns (20, 2) array of points.
    """
    n = 20
    radii = params[:n]
    angle_offsets = params[n:]
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    
    # Ensure angles are in valid range and maintain ordering for convexity
    angles = np.sort(angles)
    # Wrap around to [0, 2*pi)
    angles = angles % (2.0 * np.pi)
    angles = np.sort(angles)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _objective(params):
    points = _params_to_points(params)
    score = _polygon_score(points)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This baseline is the upstream search's starting construction: a regular polygon whose
    vertices lie evenly on a circle (deterministic, strictly convex). The scoring function
    is scale/translation invariant, so the circle radius is arbitrary.

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit
    
    n = 20
    
    # Baseline: regular polygon
    best_points = _regular_polygon_points()
    best_score = _polygon_score(best_points)
    
    # Optimization using Nelder-Mead on parameters (radii + angular offsets)
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Multiple random restarts with different noise scales
    restart_configs = [
        (0.1, 0.05),   # Small noise
        (0.3, 0.1),    # Medium noise
        (0.5, 0.15),   # Large noise
        (0.8, 0.2),    # Very large noise
    ]
    
    for r_scale, a_scale in restart_configs:
        if time.time() > deadline - 30.0:
            break
            
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize parameters
            r0 = 1.0 + r_scale * np.random.randn(n)
            a0 = a_scale * np.random.randn(n)
            # Make angular offsets small to maintain convexity
            a0 = np.clip(a0, -0.3, 0.3)
            params0 = np.concatenate([r0, a0])
            
            try:
                res = minimize(
                    _objective, 
                    params0, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 300, 'maxfev': 1500}
                )
            except Exception:
                continue
            
            r_opt = res.x
            points = _params_to_points(r_opt)
            score = _polygon_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points
                
            # If we've found a very good score, reduce remaining work
            if best_score > 1.4:
                if time.time() > deadline - 60.0:
                    break
                
    return best_points


# EVOLVE-BLOCK-END
