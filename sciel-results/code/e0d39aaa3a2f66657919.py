import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    # Generate all vectors of the form (+/-1, +/-1, 0, ..., 0)
    # These are the minimal vectors of the D11 lattice (k=220)
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Convert to array
    return np.array(points)
