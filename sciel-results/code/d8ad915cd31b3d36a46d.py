# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered, mean_dist
    return centered / mean_dist, mean_dist

def _check_strictly_convex(polygon):
    """Check if polygon is strictly convex."""
    n = len(polygon)
    cross_products = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        cross_products.append(cross)
    
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
        
    angles = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        da = a2 - a1
        while da > np.pi:
            da -= 2 * np.pi
        while da < -np.pi:
            da += 2 * np.pi
        angles.append(da)
    total_turn = sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(v_idx, polygon):
    """Compute the vertex score for vertex at v_idx in polygon."""
    n = len(polygon)
    v = polygon[v_idx]
    diffs = polygon - v
    dists = np.linalg.norm(diffs, axis=1)
    dists[v_idx] = np.inf
    other_dists = dists[~np.isinf(dists)]
    sorted_dists = np.sort(other_dists)
    
    if sorted_dists[0] < 0.05:
        return 0.0
        
    gaps = np.diff(sorted_dists)
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k + 1]) / 2
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = sorted_dists[0:4]
    elif k == 17:
        selected = sorted_dists[15:19]
    else:
        selected = sorted_dists[k - 1:k + 3]
        
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _score_polygon(polygon):
    """Compute the minimum vertex score over all vertices."""
    polygon, mean_dist = _normalize(polygon)
    if not _check_strictly_convex(polygon):
        return 0.0
    scores = [_vertex_score(i, polygon) for i in range(len(polygon))]
    return min(scores)

def _optimize_polygon(start_time, max_time=600.0):
    n = 20
    deadline = start_time + max_time - 30.0
    
    # Start with regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _score_polygon(best_polygon)
    
    # Coordinate ascent on angles with soft-min surrogate
    def compute_soft_min_score(angles, beta=50.0):
        """Compute a smooth approximation of the min vertex score."""
        base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        angles = base_angles + angles
        angles = angles - np.mean(angles)
        angles = np.sort(angles)
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        
        if np.any(np.diff(np.concatenate([angles, [angles[0] + 2*np.pi]])) < 1e-6):
            return 0.0
            
        x = np.cos(angles)
        y = np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        # Normalize
        polygon, _ = _normalize(polygon)
        
        if not _check_strictly_convex(polygon):
            return 0.0
            
        # Compute all vertex scores
        scores = np.array([_vertex_score(i, polygon) for i in range(n)])
        
        # Soft min using log-sum-exp
        if np.any(scores <= 0):
            return 0.0
        
        # soft min = -log(sum(exp(-beta * scores))) / beta
        max_score = np.max(scores)
        exp_scores = np.exp(-beta * (scores - max_score))
        soft_min = max_score - np.log(np.sum(exp_scores)) / beta
        
        return soft_min
    
    # Coordinate ascent
    np.random.seed(42)
    current_angles = np.zeros(n)
    current_score = compute_soft_min_score(current_angles)
    
    step_size = 0.05
    for iteration in range(2000):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n):
            if time.time() > deadline:
                    break
                    
            for delta in [step_size, -step_size]:
                trial_angles = current_angles.copy()
                trial_angles[i] += delta
                trial_score = compute_soft_min_score(trial_angles)
                if trial_score > current_score:
                    current_angles = trial_angles
                    current_score = trial_score
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    # Convert final angles to polygon
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    final_angles = base_angles + current_angles
    final_angles = final_angles - np.mean(final_angles)
    final_angles = np.sort(final_angles)
    final_angles = final_angles % (2 * np.pi)
    final_angles = np.sort(final_angles)
    
    final_polygon = np.stack([np.cos(final_angles), np.sin(final_angles)], axis=1)
    final_score = _score_polygon(final_polygon)
    
    if final_score > best_score:
        best_polygon = final_polygon
        best_score = final_score
    
    return best_polygon, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    polygon, score = _optimize_polygon(start_time)
    return polygon

# EVOLVE-BLOCK-END
