import numpy as np

def construct():
    p = 1009
    # Compute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    for s in range(1, 1008):
        A = 0
        B = 1009
        max_e = s - 1
        for e in range(s, 1008):
            x = e
            ix = inv[x]
            if ix < x:
                if ix > A:
                    A = ix
            elif ix > x:
                if ix < B:
                    B = ix
            
            if s > A and e < B:
                max_e = e
            else:
                break
        
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
