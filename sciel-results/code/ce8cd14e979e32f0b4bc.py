import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # Generate all points with exactly 2 nonzero coordinates, each being +1 or -1
    # This results in C(11, 2) * 4 = 55 * 4 = 220 points
    # All points have norm squared = 2
    # Minimum squared distance between any two distinct points is 2
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2 (True)
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verification
    k = len(points)
    assert k == 220
    assert points.shape == (220, 11)
    
    max_norm_sq = 0
    for p in points:
        ns = int(np.dot(p, p))
        if ns > max_norm_sq:
            max_norm_sq = ns
            
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist_sq:
                min_dist_sq = ds
                
    assert min_dist_sq >= max_norm_sq, f"min_dist={min_dist_sq} max_norm={max_norm_sq}"
    
    return points
