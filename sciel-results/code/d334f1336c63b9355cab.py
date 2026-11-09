import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries
    # Entries are from {-1, 1}
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations for (i, j)
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)
    
    return np.array(points)
