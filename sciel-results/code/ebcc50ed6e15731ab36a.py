import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # 1. Scaled E8 minimal vectors in first 8 dimensions
    # Type 1: +/- 2 e_i +/- 2 e_j
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    points.append(p)
                    
    # Type 2: (+/- 1, +/- 1, ..., +/- 1) with even number of minuses
    # 8 dimensions. 2^8 = 256 vectors. Half have even minuses.
    for mask in range(256):
        # Check if number of 1s (or minuses) is even?
        # Let's define vector v where v[i] = 1 if bit i is 0, -1 if bit i is 1.
        # We need even number of -1s.
        num_minuses = bin(mask).count('1')
        if num_minuses % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for i in range(8):
                if (mask >> i) & 1:
                    p[i] = -1
                else:
                    p[i] = 1
            points.append(p)
            
    # 2. Scaled Z^3 minimal vectors in last 3 dimensions
    # +/- 2 e_i for i=8,9,10
    for i in range(8, 11):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = 2 * s
            points.append(p)
            
    return np.array(points)
