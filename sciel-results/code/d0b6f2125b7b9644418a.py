import numpy as np

def construct():
    D = 11
    
    # Generate 240 vectors from 2 * E8
    # E8 minimal vectors:
    # 112 of type ( +/-1, +/-1, 0, ..., 0 )
    # 128 of type ( +/-1/2, ..., +/-1/2 ) with even number of minus signs
    
    vectors = []
    
    # Part 1: 2 * E8 vectors (embedded in first 8 dims)
    # Type 1: ( +/-2, +/-2, 0, ..., 0 )
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Type 2: 2 * ( +/-1/2, ..., +/-1/2 ) = ( +/-1, ..., +/-1 )
    # Condition: even number of minus signs in the original E8 vector.
    # Original: 1/2 * (x1, ..., x8) where xi = +/-1.
    # Even number of -1s.
    # Scaled: (x1, ..., x8).
    # So we need all ( +/-1, ..., +/-1 ) with even number of -1s.
    # Total 2^8 = 256. Half have even number of -1s (128).
    
    # Generate all combinations of signs
    for mask in range(256):
        v = [0] * D
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minus_count += 1
            else:
                v[i] = 1
        if minus_count % 2 == 0:
            vectors.append(v)
            
    # Part 2: Vectors in last 3 dims
    # ( +/-2, 0, 0 ) etc.
    for i in range(8, 11):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
