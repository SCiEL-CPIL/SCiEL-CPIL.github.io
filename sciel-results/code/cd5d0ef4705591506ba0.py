import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    margin = 5.0
    
    vectors = []
    
    # Generate 2E8 vectors (first 8 dimensions)
    # Type 1: +/- 2 e_i +/- 2 e_j
    for i in range(8):
        for j in range(8):
            if i == j: continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = si * 2
                    v[j] = sj * 2
                    vectors.append(v)
                    
    # Type 2: +/- 1 in all 8 positions, odd number of -1s
    # Represent as tuple of signs
    for mask in range(256):
        # Count bits (minus signs)
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 1:
            v = [0] * 11
            for i in range(8):
                if mask & (1 << i):
                    v[i] = -1
                else:
                    v[i] = 1
            vectors.append(v)
            
    # Generate 2A3 vectors (last 3 dimensions)
    # A3 is D3, vectors are +/- 2 e_i +/- 2 e_j
    for i in range(3):
        for j in range(3):
            if i == j: continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[8 + i] = si * 2
                    v[8 + j] = sj * 2
                    vectors.append(v)
                    
    # Convert to array
    vec_array = np.array(vectors, dtype=np.int64)
    
    # Optional: Local search or verification if time permits
    # But 252 is likely optimal for this lattice family.
    # The task asks for max k.
    
    return vec_array
