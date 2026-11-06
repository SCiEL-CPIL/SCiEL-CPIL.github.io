import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes p ≡ 3 (mod 4) in the range [5, 90]
    # p=7 is skipped because out-degree is 3, which is < 4
    candidates = [11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        # Generate squares for the prime field
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        # Precompute in_dom: bitmask of vertices pointing to x
        in_dom = [0] * p
        for v in range(p):
            for x in range(p):
                if v != x and squares[(x - v) % p]:
                    in_dom[x] |= (1 << v)
        
        valid = True
        # Verify every 4-set has a common dominator
        for a, b, c, d in itertools.combinations(range(p), 4):
            common = in_dom[a] & in_dom[b] & in_dom[c] & in_dom[d]
            # Exclude the vertices themselves
            common &= ~((1 << a) | (1 << b) | (1 << c) | (1 << d))
            if common == 0:
                valid = False
                break
        
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if squares[(j - i) % p]:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
    
    # Fallback to 83 if something goes wrong (should not happen if 67 works)
    p = 83
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if squares[(j - i) % p]:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
