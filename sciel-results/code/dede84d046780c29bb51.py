def construct():
    p = 1009
    # Precompute inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    left_bound = 1
    best_len = 0
    best_start = 1
    
    for e in range(1, p):
        if e == 1 or e == p - 1:
            left_bound = e + 1
        else:
            inv_e = inv[e]
            if inv_e < e:
                if left_bound <= inv_e:
                    left_bound = inv_e + 1
        
        current_len = e - left_bound + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left_bound
            
    return [best_start, best_len]
