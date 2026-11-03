import numpy as np
import itertools

def construct():
       D = 11
       vectors = []
       
       # Generate all supports of size 2
       for support in itertools.combinations(range(D), 2):
           i, j = support
           # 4 sign patterns: (1,1), (1,-1), (-1,1), (-1,-1)
           for s1 in [1, -1]:
               for s2 in [1, -1]:
                   v = np.zeros(D, dtype=np.int8)
                   v[i] = s1
                   v[j] = s2
                   vectors.append(v)
                   
       return np.array(vectors, dtype=np.int8)
