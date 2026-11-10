import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    
    # Generate E8 minimal vectors
    # Type 1: (+/-2, +/-2, 0, ..., 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
                    
    # Type 2: (+/-1, +/-1, ..., +/-1) with even number of -1s
    for mask in range(256):
        count_minus = 0
        v = np.zeros(8, dtype=np.int64)
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = -1
                count_minus += 1
            else:
                v[k] = 1
        if count_minus % 2 == 0:
            vectors.append(v)
            
    # Generate D3 minimal vectors (norm squared 2)
    # These are permutations of (+/-1, +/-1, 0)
    d3_vectors = []
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    d3_vectors.append(v)
                    
    # Combine
    # E8 vectors are in first 8 dims, D3 in last 3
    # Note: The task requires integer coordinates.
    # E8 vectors are integers. D3 vectors are integers.
    
    points = np.zeros((len(vectors) + len(d3_vectors), 11), dtype=np.int64)
    
    # Fill E8 part
    for idx, v in enumerate(vectors):
        points[idx, :8] = v
        
    # Fill D3 part
    for idx, v in enumerate(d3_vectors):
        points[len(vectors) + idx, 8:11] = v
        
    return points
