import numpy as np
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    d = 11
    points_list = []
    
    # Part 1: All vectors of type (+/-2, +/-2, 0, ...) in Z^11
    # 55 position pairs * 4 sign combos = 220 points, norm^2 = 8
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Part 2: All vectors of type (+/-1, ..., +/-1) in first 8 dims with even number of -1s
    # C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 128 points, norm^2 = 8
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points
