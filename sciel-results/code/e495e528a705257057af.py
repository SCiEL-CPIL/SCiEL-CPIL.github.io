import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 vectors:
            # 1 at i, 1 at j
            # 1 at i, -1 at j
            # -1 at i, 1 at j
            # -1 at i, -1 at j
            
            # v1: 1, 1
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            
            # v2: 1, -1
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points.append(v2)
            
            # v3: -1, 1
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            points.append(v3)
            
            # v4: -1, -1
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            points.append(v4)
            
    points_array = np.array(points, dtype=np.int64)
    return points_array
