import numpy as np
import itertools

D = 11

def construct():
    vecs = []
    
    # Type 1: Weight 1, +/- 2
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
            
    # Type 2: Weight 4, +/- 1, even sum
    # Sum is even if number of -1s is even.
    # In a 4-bit mask, if we map 0->1, 1->-1, then number of -1s is popcount(mask).
    # We need popcount(mask) to be even.
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            # Count bits
            bits = bin(mask).count("1")
            if bits % 2 == 0:
                v = np.zeros(D, dtype=np.int64)
                for j, p in enumerate(pos):
                    if (mask & (1 << j)) == 0:
                        v[p] = 1
                    else:
                        v[p] = -1
                vecs.append(v)
                
    result = np.array(vecs, dtype=np.int64)
    return result
