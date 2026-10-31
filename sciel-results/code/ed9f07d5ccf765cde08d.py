"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    vectors = []
    
    # 1. Weight-8 vectors with entries +/- 1
    # Support size 8. There are C(11, 8) = 165 supports.
    # For each support, 2^8 = 256 sign combinations.
    # Total 42240 vectors.
    # Norm squared = 8.
    
    indices = range(D)
    for support in itertools.combinations(indices, 8):
        # Generate all 256 sign combinations for this support
        # We can iterate through 0 to 255 and use bits for signs
        for mask in range(256):
            v = [0] * D
            # Set values at support indices
            # bit 0 corresponds to support[0], etc.
            for k, idx in enumerate(support):
                if (mask >> k) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            vectors.append(v)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # 2. Weight-2 vectors with entries +/- 2
    # Support size 2. There are C(11, 2) = 55 supports.
    # For each support, 4 sign combinations (++, +-, -+, --).
    # Total 220 vectors.
    # Norm squared = 2^2 + 2^2 = 8.
    
    for i, j in itertools.combinations(indices, 2):
        for si in (2, -2):
            for sj in (2, -2):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
