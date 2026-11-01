import numpy as np

D = 11

def construct():
       # Generate all vectors with exactly two non-zero entries, each +/- 1
       # There are C(11, 2) pairs of indices, and 4 sign combinations
       # Total = 55 * 4 = 220 vectors
       
       vecs = []
       for i in range(D):
           for j in range(i+1, D):
               for si in (1, -1):
                   for sj in (1, -1):
                       v = [0] * D
                       v[i] = si
                       v[j] = sj
                       vecs.append(v)
       
       return np.array(vecs, dtype=np.int64)
