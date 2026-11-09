import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    points = np.array(candidates)
    return points
