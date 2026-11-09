import numpy as np
from itertools import combinations, product
import time

def kissing_number11() -> np.ndarray:
    # The optimal known configuration for dimension 11 with the given constraints
    # is the set of roots of the lattice E8 x D3.
    # E8 has 240 roots. D3 has 12 roots.
    # Total 252 points.
    # We scale them by 2 to ensure integer coordinates and norm squared 8.
    
    points = []
    
    # Generate E8 roots scaled by 2
    # Type 1: +/- 2 e_i +/- 2 e_j
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                points.append(v)
                
    # Type 2: +/- 1 in all 8 coords, even number of -1s
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            v = [0] * 11
            for k, s in enumerate(signs):
                v[k] = s
            points.append(v)
            
    # Generate D3 roots scaled by 2
    # D3 roots are +/- e_i +/- e_j for i != j in 3D
    # Embed in last 3 dimensions (indices 8, 9, 10)
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[8 + i] = 2 * si
                v[8 + j] = 2 * sj
                points.append(v)
                
    points_arr = np.array(points, dtype=np.int64)
    return points_arr
