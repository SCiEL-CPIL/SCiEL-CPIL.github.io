import numpy as np

def kissing_number11() -> np.ndarray:
       d = 11
       points = []
       for i in range(d):
           for j in range(i + 1, d):
               # v1: +i, +j
               v1 = np.zeros(d, dtype=np.int64)
               v1[i] = 1
               v1[j] = 1
               points.append(v1)
               
               # v2: -i, -j
               v2 = np.zeros(d, dtype=np.int64)
               v2[i] = -1
               v2[j] = -1
               points.append(v2)
               
               # v3: +i, -j
               v3 = np.zeros(d, dtype=np.int64)
               v3[i] = 1
               v3[j] = -1
               points.append(v3)
               
               # v4: -i, +j
               v4 = np.zeros(d, dtype=np.int64)
               v4[i] = -1
               v4[j] = 1
               points.append(v4)
               
       return np.array(points)
