def construct():
    n = 1008
    p = 1009
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p-2, p)
    
    best_start = 2
    best_len = 0
    
    L = 2
    for R in range(2, 1008):
        inv_R = inv[R]
        if inv_R >= L:
            L = inv_R + 1
        if L <= R:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    return [best_start, best_len]
