# EVOLVE-BLOCK-START
import numpy as np
import time


def _score_polygon(pts):
    """Compute the minimum vertex score for a given polygon."""
    n = len(pts)
    pts = np.asarray(pts, dtype=np.float64)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    norm_pts = centered / mean_dist
    
    # Check strict convexity
    # No repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(norm_pts[i] - norm_pts[j]) < 1e-10:
                return 0.0
    
    # Check all turns are nonzero and of one sign
    cross_signs = []
    for i in range(n):
        p0 = norm_pts[i]
        p1 = norm_pts[(i+1) % n]
        p2 = norm_pts[(i+2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        if abs(cross) < 1e-12:
            return 0.0
        cross_signs.append(np.sign(cross))
    
    if len(set(cross_signs)) != 1:
        return 0.0
    
    # Total turning should be +-2*pi (for a simple convex polygon this is automatic
    # if all turns are same sign and it's closed, but let's verify)
    # Actually for a convex polygon with all same-sign turns, total turn is 2*pi
    # So we're good if all crosses have same sign.
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = norm_pts[i]
        # Distances from v to all other vertices
        diffs = norm_pts - v
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        # Exclude self
        dists = np.delete(dists, i)
        # Sort
        d_sorted = np.sort(dists)
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            break
        
        # Select four distances
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:  # k=17 means the last gap (between d_17 and d_18)
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
        
        # raw = mean of max(d/D, D/d) over selected
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        vertex_score = min(1.0, 1.0 / raw)
        min_score = min(min_score, vertex_score)
    
    return min_score


def _regular_polygon(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


def _local_search(initial_pts, max_iter=5000, step_size=0.05, deadline=None):
    """Local search to improve the minimum vertex score."""
    n = len(initial_pts)
    best_pts = initial_pts.copy()
    best_score = _score_polygon(best_pts)
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pts, best_score
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Generate a random perturbation
        noise = rng.normal(0, step_size, size=best_pts.shape)
        candidate = best_pts + noise
        
        # Score the candidate
        cand_score = _score_polygon(candidate)
        
        if cand_score > best_score:
            best_pts = candidate
            best_score = cand_score
            # Reduce step size slightly on improvement
            step_size *= 0.95
        else:
            # Increase step size slightly if stuck
            step_size *= 1.02
            step_size = min(step_size, 0.5)
    
    return best_pts, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum
    vertex equidistance score. Uses local search starting from a regular polygon.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin: 600s limit, use 600s for computation
    
    n = 20
    # Start from regular polygon
    initial_pts = _regular_polygon(n, radius=1.0)
    
    # First, get baseline score
    base_score = _score_polygon(initial_pts)
    
    # Try multiple local searches with different random seeds and step sizes
    best_pts = initial_pts.copy()
    best_score = base_score
    
    # Phase 1: Coarse local search
    for seed in range(10):
        if time.time() > deadline - 30.0:
            break
        rng = np.random.default_rng(seed)
        pts = initial_pts.copy()
        step_size = 0.1
        for iteration in range(2000):
            if time.time() > deadline - 30.0:
                break
            noise = rng.normal(0, step_size, size=pts.shape)
            candidate = pts + noise
            cand_score = _score_polygon(candidate)
            if cand_score > best_score:
                best_pts = candidate.copy()
                best_score = cand_score
                step_size *= 0.95
            else:
                step_size *= 1.01
                step_size = min(step_size, 0.3)
            pts = candidate
    
    # Phase 2: Fine-tuning from best found
    if time.time() < deadline - 20.0:
        pts = best_pts.copy()
        step_size = 0.02
        rng = np.random.default_rng(123)
        for iteration in range(5000):
            if time.time() > deadline - 20.0:
                break
            noise = rng.normal(0, step_size, size=pts.shape)
            candidate = pts + noise
            cand_score = _score_polygon(candidate)
            if cand_score > best_score:
                best_pts = candidate.copy()
                best_score = cand_score
                step_size *= 0.97
            else:
                step_size *= 1.005
                step_size = min(step_size, 0.1)
            pts = candidate
    
    # Return the best polygon found
    return best_pts


# EVOLVE-BLOCK-END
