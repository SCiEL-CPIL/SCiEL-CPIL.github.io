# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    # Vectorized check for min distance between any pair
    # This is O(n^2) but n=20 is small
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
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

def _params_to_points(params):
    """
    params: array of length 40 - first 20 are angle offsets, next 20 are radii.
    Returns (20, 2) array of points.
    """
    n = 20
    angle_offsets = params[:n]
    radii = params[n:]
    
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_offsets
    
    # Ensure angles are sorted to maintain cyclic order for convexity check
    # If offsets are small, they should be sorted. If not, sorting might break the "vertex i" association
    # but for convexity check, we just need the set of points to form a convex hull in order.
    # However, the score depends on the specific vertex i having specific distances.
    # If we sort angles, the indices change. 
    # Strategy: Keep indices fixed. If angles cross, the polygon is self-intersecting or non-convex.
    # The validity check will catch non-convexity.
    # To help the optimizer, we can wrap angles to [-pi, pi] relative to base?
    # No, just use them as is. The DE bounds will keep them close enough to avoid wrapping issues mostly.
    
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective(params):
    points = _params_to_points(params)
    score = _polygon_score(points)
    # Return negative score for minimization
    # If score is 0 (invalid), return a large penalty to guide search away
    if score == 0.0:
        return 100.0
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
    deadline = start_time + 600.0  # Safety margin before 630s limit
    
    n = 20
    
    # Baseline: regular polygon
    best_points = _params_to_points(np.concatenate([np.zeros(n), np.ones(n)]))
    best_score = _polygon_score(best_points)
    best_params = np.concatenate([np.zeros(n), np.ones(n)])
    
    # 1. Differential Evolution for Global Search
    # Parameters: 20 angle offsets, 20 radii
    # Bounds for angle offsets: small enough to likely maintain convexity/order, but large enough to explore
    # pi/20 is the gap. Let's allow +/- 0.4 * gap to allow some distortion but not crossing
    angle_bound = 0.4 * (2 * np.pi / n)
    radius_bounds = (0.5, 1.5)
    
    bounds = [( -angle_bound, angle_bound)] * n + [radius_bounds] * n
    
    try:
        # Use DE to find a good starting point
        # popsize=15, maxiter=100, tol=1e-5
        # This is a global optimizer, good for escaping local minima
        res_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False # We will polish manually
        )
        
        if res_de.success or res_de.fun < _objective(best_params):
            if res_de.fun < 100.0: # Valid score found
                current_params = res_de.x
                current_score = -res_de.fun
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params
                    best_points = _params_to_points(current_params)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60.0:
        return best_points

    # 2. Nelder-Mead Refinement
    # Run local optimization from the best point found by DE (or initial if DE failed)
    # We can run multiple Nelder-Mead restarts from slightly perturbed versions of the best solution
    
    max_refinements = 5
    for i in range(max_refinements):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb the best params slightly to escape potential local minima of NM
        if i == 0:
            current_params = best_params.copy()
        else:
            current_params = best_params + 0.01 * np.random.randn(2*n)
            
        try:
            res_nm = minimize(
                _objective, 
                current_params, 
                method='Nelder-Mead', 
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 2000, 'maxfev': 5000, 'adaptive': True}
            )
            
            if res_nm.fun < 100.0: # Valid
                nm_score = -res_nm.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_params = res_nm.x
                    best_points = _params_to_points(best_params)
        except Exception:
            continue
            
    return best_points

# EVOLVE-BLOCK-END
