def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 1
    
    for L in range(1, N + 1):
        # Check if L itself is self-inverse
        if inv[L] == L:
            # L cannot be in any valid interval of length >= 1
            continue
        valid = True
        for R in range(L, N + 1):
            if not valid:
                break
            # Check if adding R makes the window invalid
            if L <= inv[R] <= R:
                valid = False
                break
            # Window [L, R] is valid
            if R - L + 1 > best_len:
                best_len = R - L + 1
                best_start = L
    
    return [best_start, best_len]
