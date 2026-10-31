import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    vectors = []
    
    # 1. Generate scaled E8 minimal vectors in R^8
    # Part A: Weight-2 vectors (from D8)
    # All pairs (i, j) with i < j, signs s1, s2 in {1, -1}
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * 8
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v + [0, 0, 0]) # Pad to 11
    
    # Part B: Weight-8 vectors (from half-integers)
    # All vectors in {1, -1}^8 with sum divisible by 4
    # Sum = 2k - 8. 2k - 8 = 4m => 2k = 4m + 8 => k = 2m + 4.
    # k must be even.
    for signs in itertools.product([1, -1], repeat=8):
        s = sum(signs)
        if s % 4 == 0:
            vectors.append(list(signs) + [0, 0, 0]) # Pad to 11
    
    # 2. Generate D3 minimal vectors in R^3
    # All weight-2 vectors in R^3
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * 3
                    v[i] = s1
                    v[j] = s2
                    vectors.append([0] * 8 + v) # Pad to 11
    
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    return result
