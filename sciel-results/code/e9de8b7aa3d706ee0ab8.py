import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # 1. Generate E8 minimal vectors scaled by 2
    # Type 1: (1, 1, 0, ...) permutations
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i], v[j] = si * 2, sj * 2
                vectors.append(v)
                
    # Type 2: (1/2, ...) scaled by 2 -> (1, ...)
    # All coords are +/- 1. Sum of original coords was even.
    # Original: sum of 8 half-integers is even.
    # Scaled: sum of 8 integers is even.
    # So we need vectors in {1, -1}^8 with even sum.
    # Sum of 8 numbers is even if there are an even number of -1s?
    # Sum = (number of 1s) - (number of -1s).
    # Let k be number of -1s. Sum = (8-k) - k = 8 - 2k.
    # 8 - 2k is always even.
    # So ALL 2^8 = 256 vectors in {1, -1}^8 have even sum?
    # Wait. The condition for E8 is that the sum of coordinates is an even INTEGER.
    # For half-integers, sum is integer.
    # (1/2 + ... + 1/2) = 4. Even.
    # (1/2 + ... - 1/2 + 1/2) = 3. Odd.
    # So we need the sum of the SIGNED half-integers to be even.
    # This is equivalent to the number of minus signs being even?
    # Sum = 4 - (number of minus signs).
    # If number of minus signs is even, sum is even.
    # If number of minus signs is odd, sum is odd.
    # So we take vectors with an EVEN number of minus signs.
    
    for bits in itertools.product([1, -1], repeat=8):
        # Count minus signs
        if sum(1 for b in bits if b == -1) % 2 == 0:
            vectors.append(list(bits))
            
    # Now we have 240 vectors in Z^8.
    # Pad to 11
    for v in vectors:
        v.extend([0, 0, 0])
        
    # 2. Generate Z^3 minimal vectors
    z3_vectors = []
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                w = [0] * 3
                w[i], w[j] = si, sj
                z3_vectors.append(w)
                
    for w in z3_vectors:
        w_vec = [0] * 8 + w
        vectors.append(w_vec)
        
    return np.array(vectors, dtype=np.int64)
