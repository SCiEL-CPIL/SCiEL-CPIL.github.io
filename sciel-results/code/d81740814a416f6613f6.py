# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their maximum squared norm is smaller than or equal to their minimum pairwise squared distance.
    
    This implementation uses the set of all integer points in R^11 with squared norm 2.
    These points have exactly two non-zero coordinates, which are +/-1.
    There are C(11, 2) * 2^2 = 55 * 4 = 220 such points.
    
    For any two distinct points u, v in this set:
    - max squared norm is 2.
    - min squared distance is 2 (occurs when they share one index with the same sign, 
      or are identical in one index and differ in the other, etc. specifically 
      if they share an index with same sign, the distance squared is 1^2 + 1^2 = 2).
    
    Since min_dist_sq (2) >= max_norm_sq (2), the condition is satisfied.
    """
    start_time = time.time()
    deadline = 600.0
    
    d = 11
    points = []
    
    # Generate all points with exactly two non-zero coordinates +/-1
    for i in range(d):
        for j in range(i + 1, d):
            # Signs for i and j can be +/-1
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    points = np.array(points, dtype=np.int64)
    
    # Verify condition
    # Max norm squared
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Min pairwise distance squared
    min_dist_sq = float('inf')
    n_points = len(points)
    
    # Optimization: We know the answer is 2, but we verify to be safe.
    # 220 points -> 220*219/2 = 24090 pairs. This is fast.
    for i in range(n_points):
        for j in range(i + 1, n_points):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            # Early exit if we find a violation? No, we just need min.
            # But if min_dist_sq < max_norm_sq, it's invalid.
            # We can break early if min_dist_sq < max_norm_sq? 
            # No, we need the exact min to be sure it's >= max.
            # But we know it's 2.
            
    if min_dist_sq >= max_norm_sq and n_points >= 2:
        return points
    else:
        # Fallback to a smaller valid set (e.g., norm 1 points)
        # Norm 1 points: +/- e_i. 22 points.
        # Min dist sq 2. Max norm sq 1. 2 >= 1. Valid.
        fallback_points = []
        for i in range(d):
            for s in [-1, 1]:
                p = [0] * d
                p[i] = s
                fallback_points.append(p)
        return np.array(fallback_points, dtype=np.int64)

# EVOLVE-BLOCK-END
