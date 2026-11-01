import numpy as np
def construct():
    D = 11
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # e_i - e_j
            v1 = [0] * D
            v1[i] = 1
            v1[j] = -1
            vectors.append(v1)
            
            # -e_i + e_j
            v2 = [0] * D
            v2[i] = -1
            v2[j] = 1
            vectors.append(v2)
            
            # e_i + e_j
            v3 = [0] * D
            v3[i] = 1
            v3[j] = 1
            vectors.append(v3)
            
            # -e_i - e_j
            v4 = [0] * D
            v4[i] = -1
            v4[j] = -1
            vectors.append(v4)
            
    return np.array(vectors, dtype=np.int64)
