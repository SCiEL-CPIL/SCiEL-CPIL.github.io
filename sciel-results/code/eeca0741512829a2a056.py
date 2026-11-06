import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 80.0  # Safety margin
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    left = 1
    
    for right in range(1, MOD):
        if time.time() > deadline:
            break
            
        ix = inv[right]
        
        # If right is a self-inverse, it cannot be in the interval
        if ix == right:
            left = right + 1
        # If the inverse is within the current window [left, right-1],
        # we must move left past it
        elif left <= ix < right:
            left = ix + 1
            
        # Calculate current valid interval length
        curr_len = right - left + 1
        if curr_len > best_len:
            best_len = curr_len
            best_start = left
            
    # Fallback
    if best_len == 0:
        for s in range(1, MOD):
            if inv[s] != s:
                best_start = s
                best_len = 1
                break
    if best_len == 0:
        best_start = 1
        best_len = 1

    return [best_start, best_len]

# EVOLVE-BLOCK-END
