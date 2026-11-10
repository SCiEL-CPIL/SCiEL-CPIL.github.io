import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs 220 integer points in 11 dimensions (D11 lattice minimal vectors)
    such that min pairwise squared distance >= max squared norm.
    
    Returns:
        points: np.ndarray of shape (220, 11) with dtype int64
    """
    d = 11
    points = []
    
    # Generate all vectors of form ±e_i ± e_j for i < j
    for i in range(d):
        for j in range(i + 1, d):
            # +e_i + e_j
            v = [0] * d
            v[i] = 1
            v[j] = 1
            points.append(v)
            # +e_i - e_j
            v = [0] * d
            v[i] = 1
            v[j] = -1
            points.append(v)
            # -e_i + e_j
            v = [0] * d
            v[i] = -1
            v[j] = 1
            points.append(v)
            # -e_i - e_j
            v = [0] * d
            v[i] = -1
            v[j] = -1
            points.append(v)
    
    points = np.array(points, dtype=np.int64)
    
    # Verify validity
    if len(points) < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    norms = np.sum(points**2, axis=1)
    max_norm = np.max(norms)
    
    min_dist_sq = float('inf')
    k = len(points)
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.dot(diff, diff))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm:
        # Fallback: return just 2 points
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return points
