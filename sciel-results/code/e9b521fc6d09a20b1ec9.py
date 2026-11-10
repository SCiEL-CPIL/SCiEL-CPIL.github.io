import time

def construct():
    deadline = time.time() + 88.0
    
    p = 1009
    # Precompute modular inverses
    # inv[x] = x^{-1} mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    max_len = 0
    best_start = 1
    
    # We iterate over all possible start positions
    for s in range(1, p):
        if time.time() > deadline:
            break
            
        # If s is a self-inverse, the interval [s, s] is invalid because s*s % 1009 == 1
        if s == 1 or s == p - 1:
            continue
            
        current_elements = set()
        # Expand interval to the right
        for e in range(s, p):
            if time.time() > deadline:
                break
                
            # Check if e is self-inverse
            if e == 1 or e == p - 1:
                break
                
            # Check if inverse of e is already in the interval
            if inv[e] in current_elements:
                break
                
            # e is safe to add
            current_elements.add(e)
            length = e - s + 1
            if length > max_len:
                max_len = length
                best_start = s
                
    return [best_start, max_len]
