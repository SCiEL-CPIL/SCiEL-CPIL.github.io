import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of minimal vectors of the D11 lattice (points with exactly two non-zero coordinates +/-1).
    There are 2 * 11 * 10 = 220 such points.
    Max squared norm is 2.
    Min squared distance is 2.
    Condition min_dist >= max_norm holds (2 >= 2).
    """
    d = 11
    points = []
    
    # Generate all pairs of indices
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations for coordinates i and j
            # (1, 1), (1, -1), (-1, 1), (-1, -1)
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    # Convert to numpy array
    points_array = np.array(points)
    
    # Optional: Verification
    # k = len(points_array)
    # if k >= 2:
    #     norms = np.sum(points_array**2, axis=1)
    #     max_norm = np.max(norms)
    #     min_dist = float('inf')
    #     for a in range(k):
    #         for b in range(a+1, k):
    #             diff = points_array[a] - points_array[b]
    #             dist_sq = np.sum(diff**2)
    #             if dist_sq < min_dist:
    #                 min_dist = dist_sq
    #     if min_dist < max_norm:
    #         # Fallback to seed if somehow invalid (should not happen)
    #         return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    
    return points_array
