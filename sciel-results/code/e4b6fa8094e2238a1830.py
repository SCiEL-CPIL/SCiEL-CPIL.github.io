import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Primes p = 3 mod 4
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_result = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Generate Paley tournament
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        # in_mask[v] will have bit u set if u -> v
        in_masks = [0] * p
        
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    orientation.append(0)
                    in_masks[j] |= (1 << i)
                else:
                    # j -> i
                    orientation.append(1)
                    in_masks[i] |= (1 << j)
        
        # Check validity
        valid = True
        # Iterate through all 4-subsets
        # Optimization: Precompute masks for single vertices to exclude?
        # mask_out = ~(1<<a | 1<<b | 1<<c | 1<<d)
        
        # To speed up, we can break early on first violation
        for a, b, c, d in itertools.combinations(range(p), 4):
            if time.time() > deadline:
                valid = False
                break
                
            common_doms = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
            # Exclude a, b, c, d
            common_doms &= ~(1 << a | 1 << b | 1 << c | 1 << d)
            
            if common_doms == 0:
                valid = False
                break
                
        if valid:
            best_result = {'n': p, 'orientation': orientation}
            # Since we iterate from smallest p, the first valid one is the best
            return best_result
            
    # If no smaller one found, return the seed's 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
