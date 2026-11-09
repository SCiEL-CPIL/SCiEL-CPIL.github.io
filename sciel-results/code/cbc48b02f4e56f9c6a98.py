# EVOLVE-BLOCK-START
import numpy as np
import time


def _vertex_score(polygon):
    """
    Compute the per-vertex score for a convex polygon as defined in the task contract.
    Returns an array of shape (n,).
    """
    n = len(polygon)
    scores = np.zeros(n)
    
    for i in range(n):
        v = polygon[i]
        # Compute distances to all other vertices
        d = np.linalg.norm(polygon - v, axis=1)
        d[i] = np.inf  # Exclude self
        
        # Sort the 19 distances
        d_sorted = np.sort(d)
        
        # Check for any distance < 0.05
        if np.any(d_sorted < 0.05):
            scores[i] = 0.0
            continue
            
        # Find k: first index minimizing d_{k+1} - d_k
        # Note: d_sorted has 19 elements (indices 0..18)
        # Gaps are d_sorted[j+1] - d_sorted[j] for j in 0..17
        gaps = np.diff(d_sorted)
        
        # If all gaps are 0, k=0 is the first minimum
        min_gap = np.min(gaps)
        k = np.argmin(gaps)
        
        # D = (d_k + d_{k+1})/2
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        # Determine the 4 distances to use
        if k == 0:
            ds = d_sorted[0:4]
        elif k == 17:
            ds = d_sorted[15:19]
        else:
            ds = d_sorted[k-1:k+3]
            
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(ds / D, D / ds)
        raw = np.mean(ratios)
        
        # vertex score = min(1, 1/raw)
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
    return scores


def _is_convex(polygon):
    """
    Check if the polygon is strictly convex.
    """
    n = len(polygon)
    if n < 3:
        return False
        
    # Check for repeated points
    for i in range(n):
        if np.linalg.norm(polygon[i] - polygon[(i+1)%n]) < 1e-10:
            return False
            
    # Check turn directions
    turns = np.zeros(n)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1)%n]
        p2 = polygon[(i+2)%n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Cross product z-component
        turn = v1[0]*v2[1] - v1[1]*v2[0]
        turns[i] = turn
        
    # All turns must be non-zero and same sign
    if np.any(np.abs(turns) < 1e-10):
        return False
        
    if np.all(turns > 0) or np.all(turns < 0):
        return True
    else:
        return False


def _normalize_polygon(polygon):
    """
    Normalize the polygon: subtract centroid, divide by mean distance to centroid.
    """
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    
    distances = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(distances)
    
    if mean_dist <= 0.01:
        return None, 0.0
        
    polygon = polygon / mean_dist
    
    return polygon, mean_dist


def _total_turning(polygon):
    """
    Compute total turning angle. Should be +/- 2*pi for a simple convex polygon.
    """
    n = len(polygon)
    total = 0.0
    
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1)%n]
        p2 = polygon[(i+2)%n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Angle between v1 and v2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Determine sign using cross product
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross > 0:
            total += angle
        elif cross < 0:
            total -= angle
            
    return total


def _score_polygon(polygon):
    """
    Compute the final metric score for a polygon.
    """
    # Normalize
    polygon_norm, mean_dist = _normalize_polygon(polygon)
    
    if polygon_norm is None:
        return 0.0
        
    # Check convexity
    if not _is_convex(polygon_norm):
        return 0.0
        
    # Check total turning
    total_turn = _total_turning(polygon_norm)
    if not np.isclose(abs(total_turn), 2*np.pi, atol=1e-4):
        return 0.0
        
    # Compute vertex scores
    scores = _vertex_score(polygon_norm)
    
    # Metric is the minimum vertex score
    return np.min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Initialize with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 1.0
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    # Evaluate initial score
    best_polygon = polygon.copy()
    best_score = _score_polygon(polygon)
    
    # Local search: perturb angles slightly
    step_size = 0.01
    max_iterations = 10000
    
    for iter in range(max_iterations):
        if time.time() > deadline - 10:  # Safety margin
            break
            
        # Random perturbation
        angles_perturbed = angles.copy()
        mask = np.random.rand(n) < 0.3  # Perturb 30% of vertices
        angles_perturbed[mask] += (np.random.rand(np.sum(mask)) - 0.5) * step_size
        
        # Ensure angles are still in [0, 2*pi) and sorted
        angles_perturbed = np.sort(angles_perturbed % (2*np.pi))
        
        x_pert = radius * np.cos(angles_perturbed)
        y_pert = radius * np.sin(angles_perturbed)
        polygon_pert = np.stack([x_pert, y_pert], axis=1)
        
        score_pert = _score_polygon(polygon_pert)
        
        if score_pert > best_score:
            best_score = score_pert
            best_polygon = polygon_pert
            # Reduce step size on improvement
            step_size *= 0.9
        else:
            # Gradually reduce step size
            step_size *= 0.995
        
        # If step size is too small, break
        if step_size < 1e-8:
            break
    
    return best_polygon


# EVOLVE-BLOCK-END
