def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse: x^2 = 1 mod p => x=1 or x=p-1=1008
    # So valid range is [2, 1007]
    
    # partner[x] = inv[x] for x in [2, 1007]
    # Every x in [2,1007] has partner in [2,1007] and partner[x] != x
    
    a = 2
    best_start = 2
    best_len = 0
    
    for b in range(2, 1008):  # b from 2 to 1007
        # Add b to window [a, b]
        # Check if partner[b] is in [a, b]
        pb = inv[b]
        if a <= pb <= b:
            # Conflict: need a > pb
            a = pb + 1
        
        # Now [a, b] should be valid
        length = b - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
