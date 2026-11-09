import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    # Use 4-set based vectors: each vector has 4 non-zero ±1 entries, norm_sq=4
    # Constraint: no two selected 4-sets share 3+ indices (ensures min dist_sq >= 4)
    # For each valid 4-set, use all 16 sign combinations
    
    all_4sets = list(combinations(range(d), 4))
    selected_4sets = []
    
    for cs in all_4sets:
        if time.time() > deadline - 60:
            break
        valid = True
        for existing in selected_4sets:
            if len(set(cs) & set(existing)) >= 3:
                valid = False
                break
        if valid:
            selected_4sets.append(cs)
    
    # For each selected 4-set, add all 16 sign vectors
    points = []
    for cs in selected_4sets:
        for signs in product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(cs, signs):
                v[idx] = s
            points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array
