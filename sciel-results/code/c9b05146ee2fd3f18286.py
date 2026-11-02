import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    D = 11
    
    vectors = []
    
    # 1. Generate D8 roots (scaled by 2)
    # D8 roots: exactly two non-zero entries, each +/- 1.
    # Scaled: exactly two non-zero entries, each +/- 2.
    # Dimensions 0..7
    
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (2, -2):
                for sj in (2, -2):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    # 2. Generate E8 half-integer roots (scaled by 2)
    # Original: 8 entries of +/- 0.5, even number of minuses.
    # Scaled: 8 entries of +/- 1, even number of minuses.
    # Dimensions 0..7
    
    # Generate all 8-bit sign patterns
    # Even number of -1s.
    # Total 128 patterns.
    
    # We can iterate over all 256 patterns and filter, or generate combinatorially.
    # 256 is small.
    
    for signs in itertools.product([1, -1], repeat=8):
        # Count number of -1s
        if signs.count(-1) % 2 == 0:
            v = [0] * D
            for k in range(8):
                v[k] = signs[k]
            vectors.append(v)
            
    # Total vectors should be 112 + 128 = 240
    
    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Verification
    # But 240 is small, O(N^2 * D) is fast.
    if len(result) > 1:
        norms_sq = np.array([np.dot(v, v) for v in result])
        max_ns = int(np.max(norms_sq))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                diff = result[i] - result[j]
                if np.dot(diff, diff) < max_ns:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to D11 roots if something went wrong (shouldn't happen)
            d11_vecs = []
            for i in range(D):
                for j in range(i + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            d11_vecs.append(v)
            return np.array(d11_vecs, dtype=np.int64)
            
    return result
