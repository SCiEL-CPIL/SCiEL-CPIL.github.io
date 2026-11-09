import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # 1. All 220 vectors of type (+/-2, +/-2, 0, ...) in Z^11
    # These are 2 * (roots of D_11)
    for i, j in combinations(range(11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
                
    # 2. All 128 vectors of type (+/-1, ..., +/-1) in first 8 dims
    # These are 2 * (Type 2 roots of E_8)
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)
            
    points = np.array(points_list, dtype=np.int64)
    
    # Verification
    norms = np.sum(points**2, axis=1)
    assert np.all(norms == 8), f"Not all norms are 8: {np.unique(norms)}"
    
    min_dist_sq = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                if min_dist_sq < 8:
                    break
        if min_dist_sq < 8:
            break
            
    assert min_dist_sq >= 8, f"Min dist^2 = {min_dist_sq} < 8"
    
    return points
