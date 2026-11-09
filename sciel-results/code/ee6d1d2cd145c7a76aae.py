import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of all vectors of the form ±e_i ± e_j for 1 <= i < j <= 11.
    This set has 220 points, all with squared norm 2.
    The minimum squared distance between any pair is 2.
    The maximum squared norm is 2.
    2 >= 2, so the condition is satisfied.
    """
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 sign combinations for e_i and e_j
            # 1: +e_i +e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = 1
            points.append(v)
            
            # 2: +e_i -e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            points.append(v)
            
            # 3: -e_i +e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = 1
            points.append(v)
            
            # 4: -e_i -e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = -1
            points.append(v)
            
    points = np.array(points)
    return points
