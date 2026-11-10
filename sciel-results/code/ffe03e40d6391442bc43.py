import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate E8 minimal vectors scaled by 2
    # Type 1: (±2, ±2, 0, ..., 0)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [-2, 2]:
                for s2 in [-2, 2]:
                    vec = np.zeros(d, dtype=np.int64)
                    vec[i] = s1
                    vec[j] = s2
                    points.append(vec)
                    
    # Type 2: (±1, ±1, ..., ±1) with even number of -1s
    # There are 2^8 = 256 sign vectors. Half have even number of -1s.
    for mask in range(1 << 8):
        # Count number of -1s (or 1s in mask, let's say 1 in mask means -1)
        # If we define bit 1 as -1 and 0 as 1:
        # Number of -1s is popcount(mask)
        # We need even popcount
        if bin(mask).count("1") % 2 == 0:
            vec = np.ones(d, dtype=np.int64)
            for i in range(8):
                if (mask >> i) & 1:
                    vec[i] = -1
                else:
                    vec[i] = 1
            # Indices 8, 9, 10 remain 1? 
            # Wait, E8 is in 8D. 
            # The vector is in R^8. 
            # When embedding in R^11, indices 8,9,10 should be 0.
            # My initialization `np.ones(d)` sets indices 8,9,10 to 1.
            # I need to set them to 0.
            vec[8:] = 0
            points.append(vec)
            
    # Verify count: 112 + 128 = 240
    # Convert to array
    return np.array(points, dtype=np.int64)
