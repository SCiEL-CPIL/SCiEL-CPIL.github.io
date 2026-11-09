# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs 220 points in Z^11 of the form (±2, ±2, 0, ..., 0) with two
    non-zero entries. All points have squared norm 8, and the minimum pairwise
    squared distance is 8, satisfying the kissing configuration constraint.
    
    Returns:
        points: np.ndarray of shape (220, 11)
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    points_list = []
    
    # All weight-2 points: (±2, ±2, 0, ..., 0) in 11 dimensions
    # C(11,2) * 4 = 55 * 4 = 220 points
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the solution
    k = points.shape[0]
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Check no zero vector
    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        norms_sq = np.sum(points**2, axis=1)
    
    # Check pairwise distances
    min_dist_sq = float('inf')
    max_norm_sq = int(np.max(norms_sq))
    
    valid = True
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break
        if time.time() > deadline:
            break
    
    # Check validity
    valid = valid and (min_dist_sq >= max_norm_sq)
    
    if not valid:
        # Fallback: 2-point solution
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = np.around(points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
