def construct():
    p = 1009
    # Precompute modular inverses for all values 1..1008
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        # If we can't beat best_len even extending to p-1=1008, stop
        if (p - 1) - s + 1 <= best_len:
            break
        
        # Extend interval from s as far right as possible
        e = s
        valid = True
        while e < p:
            # Check diagonal: e^2 == 1 mod p => e==1 or e==p-1
            if e == 1 or e == p - 1:
                break
            
            # Check off-diagonal: inv(e) must not be in [s, e-1]
            # Since inv(e) < e is not guaranteed, check if inv(e) is in [s, e-1]
            if s <= inv[e] <= e - 1:
                break
            
            e += 1
        
        current_len = e - s
        # e is the first invalid position, so interval is [s, e-1] with length e-s
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    return [best_start, best_len]
