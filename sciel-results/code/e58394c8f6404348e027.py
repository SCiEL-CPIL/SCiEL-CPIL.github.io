import numpy as np
from itertools import combinations, product
import time

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 540
    
    # Generate E8 vectors scaled by 2
    # Norm squared will be 8
    e8_vectors = []
    
    # Weight 2: +/- e_i +/- e_j
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
                
    # Weight 8: +/- 1 in all 8 coords, even number of -1s
    # Original E8 weight 8 vectors are +/- 1/2. Scaled by 2 -> +/- 1.
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            e8_vectors.append(list(signs))
            
    # Pad E8 to 11D
    e8_points = []
    for v in e8_vectors:
        vec = v + [0, 0, 0]
        e8_points.append(vec)
        
    # Generate D3 vectors scaled by 2
    # Norm squared will be 8
    d3_vectors = []
    # D3 minimal vectors are +/- e_i +/- e_j for i != j in 3D
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 3
                v[i] = 2 * si
                v[j] = 2 * sj
                d3_vectors.append(v)
                
    # Pad D3 to 11D (first 8 are 0)
    d3_points = []
    for v in d3_vectors:
        vec = [0] * 8 + v
        d3_points.append(vec)
        
    all_points = e8_points + d3_points
    points_arr = np.array(all_points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    return points_arr
