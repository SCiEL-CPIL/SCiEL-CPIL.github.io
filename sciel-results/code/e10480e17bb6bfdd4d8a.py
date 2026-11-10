import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all minimal vectors of the D11 lattice
    # These are vectors with exactly two non-zero coordinates, which are +/-1.
    # The sum of coordinates must be even, which is satisfied by +/-1, +/-1.
    
    for i in range(d):
        for j in range(i + 1, d):
            # 4 combinations of signs for the two non-zero coordinates
            # +1, +1
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            
            # +1, -1
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points.append(v2)
            
            # -1, +1
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            points.append(v3)
            
            # -1, -1
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            points.append(v4)
            
    points_arr = np.array(points, dtype=np.int64)
    
    # The task requires:
    # 1. 2-D array of shape (k, 11)
    # 2. Entries rounded with np.around and cast to int64
    # 3. No point is zero vector
    # 4. Min squared distance >= Max squared norm
    
    # Our construction satisfies these.
    # Max squared norm is 2.
    # Min squared distance is 2.
    # 2 >= 2 is True.
    
    return points_arr
