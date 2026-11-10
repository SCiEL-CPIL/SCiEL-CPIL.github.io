import numpy as np

def calculate_score(angles):
    n = len(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.stack([x, y], axis=1)
    
    # Check convexity
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross <= 0:
            return 0.0
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    min_score = 1.0
    for i in range(n):
        vi = pts[i]
        dists_i = np.linalg.norm(pts - vi, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists_i = dists_i[mask]
        
        if np.min(dists_i) < 0.05:
            return 0.0
        
        sorted_dists = np.sort(dists_i)
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17:
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = sorted_dists[idxs]
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    best_deltas = np.full(n, 2.0 * np.pi / n)
    
    def get_angles(deltas):
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = np.sum(deltas[:i])
        return angles
    
    best_score = calculate_score(get_angles(best_deltas))
    
    rng = np.random.default_rng(42)
    step_size = 0.05
    no_improve = 0
    
    for iteration in range(2000):
        if no_improve > 200:
            step_size *= 0.7
            no_improve = 0
            if step_size < 0.001:
                break
        
        deltas = best_deltas * (1 + rng.normal(0, step_size, n))
        deltas = np.clip(deltas, 0.01, None)
        deltas = deltas / np.sum(deltas) * 2 * np.pi
        
        angles = get_angles(deltas)
        score = calculate_score(angles)
        
        if score > best_score:
            best_score = score
            best_deltas = deltas
            no_improve = 0
        else:
            no_improve += 1
    
    angles = get_angles(best_deltas)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)
