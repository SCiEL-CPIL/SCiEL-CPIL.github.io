import numpy as np

def construct():
    D = 11
    vectors = []
    
    # Generate E8 minimal vectors (scaled by 2 to be integers with norm 2*sqrt(2))
    # Type 1: Two non-zero entries +/- 1, rest 0. Scaled by 2 -> +/- 2.
    # There are C(8,2) * 4 = 112 such vectors.
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Type 2: All 8 entries +/- 1, even number of minus signs. Scaled by 2 -> +/- 2.
    # There are 2^8 / 2 = 128 such vectors.
    # Iterate through all 256 combinations of +/- 1
    for mask in range(256):
        v = [0] * 8
        num_minus = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -2
                num_minus += 1
            else:
                v[i] = 2
        if num_minus % 2 == 0:
            vectors.append(v)
            
    # Pad to dimension 11
    vectors = [v + [0, 0, 0] for v in vectors]
    
    return np.array(vectors, dtype=np.int64)
