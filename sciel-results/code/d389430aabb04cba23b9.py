import time

def construct():
    p = 1009
    n = 1008
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Identify self-inverse elements (x*x % p == 1)
    # These are 1 and p-1 = 1008
    is_self_inv = [False] * (p)
    for x in range(1, p):
        if inv[x] == x:
            is_self_inv[x] = True
    
    # Two-pointer sliding window approach:
    # For each right endpoint R, find the minimum valid left endpoint L
    # such that [L, R] contains no inverse pairs and no self-inverses.
    #
    # A window [L, R] is invalid if:
    #   1. It contains a self-inverse element (1 or 1008)
    #   2. It contains both x and inv[x] for some x
    #
    # We track conflicts: when we add element R to the window, a conflict
    # arises if R is self-inverse, or if inv[R] is already in the window.
    
    best_start = 1
    best_len = 1
    
    L = 1
    present = [0] * (p)  # count of each residue in current window
    conflict_count = 0   # number of conflicting elements/pairs in window
    
    for R in range(1, p):  # R from 1 to 1008
        # Add R to window
        present[R] += 1
        if is_self_inv[R]:
            conflict_count += 1
        else:
            pr = inv[R]
            if pr >= 1 and pr <= n and present[pr] > 0:
                conflict_count += 1
        
        # Shrink L until window is valid
        while conflict_count > 0 and L <= R:
            # Remove L from window
            if is_self_inv[L]:
                conflict_count -= 1
            else:
                pr = inv[L]
                if pr >= 1 and pr <= n and present[pr] > 0:
                    conflict_count -= 1
            present[L] -= 1
            L += 1
        
        # Now [L, R] is valid
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
        
        # Time guard
        if time.time() > deadline:
            break
    
    # Verify the best answer
    start = best_start
    end = best_start + best_len - 1
    valid = True
    for x in range(start, end + 1):
        ix = inv[x]
        if ix == x:
            valid = False
            break
        if start <= ix <= end:
            valid = False
            break
    
    if valid:
        return [best_start, best_len]
    
    # Fallback: try a few known-good small intervals
    for s in [2, 3, 4, 5]:
        for l in range(1, 40):
            e = s + l - 1
            if e > n:
                break
            ok = True
            for x in range(s, e + 1):
                ix = inv[x]
                if ix == x or (s <= ix <= e):
                    ok = False
                    break
            if ok:
                return [s, l]
    
    return [2, 1]
