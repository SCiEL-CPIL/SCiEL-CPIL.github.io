import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors in Z^11 with squared norm 2
    # These have exactly two non-zero coordinates, each being +1 or -1
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
                    
    # Convert to array
    arr = np.array(points, dtype=np.int64)
    
    # The task requires rounding and casting, though they are already integers
    arr = np.around(arr).astype(np.int64)
    
    return arr
