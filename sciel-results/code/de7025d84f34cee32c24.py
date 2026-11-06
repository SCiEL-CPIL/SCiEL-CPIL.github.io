"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_length = 0
    
    min_start = 1
    
    for e in range(1, n + 1):
        if time.time() > deadline:
            break
            
        ix = inv[e]
        
        if ix == e:
            # Self-inverse, cannot be in any valid interval
            min_start = e + 1
        elif ix < e:
            # inv[e] is in [1, e-1], so start must be > inv[e]
            if min_start <= ix:
                min_start = ix + 1
        else:
            # inv[e] > e, no new constraint on start from this element
            pass
        
        if min_start <= e:
            length = e - min_start + 1
            if length > best_length:
                best_length = length
                best_start = min_start
        else:
            # No valid interval ending at e
            pass

    if best_length <= 0:
        # Fallback: find any valid single element
        # This should not happen for p=1009, but good for safety
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
    
    return [best_start, best_length]
