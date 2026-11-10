import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    
    points = []
    for xa in range(p):
        for xb in range(p):
            x_idx = xa + xb * p
            for ya in range(p):
                for yb in range(p):
                    y_idx = ya + yb * p
                    # S is y = 0
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
                    
    return np.array(points, dtype=np.int64)
