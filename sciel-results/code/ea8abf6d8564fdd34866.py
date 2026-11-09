def construct():
    MOD = 1009
    N = 1008
    
    # Compute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Valid interval must be within [2, 1007] (exclude self-inverse 1 and 1008)
    lo = 2
    hi = 1007
    
    best_start = lo
    best_len = 0
    
    left = lo
    for right in range(lo, hi + 1):
        inv_r = inv[right]
        # inv_r is in [1, 1008]. For x in [2,1007], inv[x] != x.
        # Check if inv_r is in the current window [left, right]
        if lo <= inv_r <= hi and inv_r >= left and inv_r < right:
            # Conflict detected: both right and inv_r would be in the window
            left = inv_r + 1
        
        if left < lo:
            left = lo
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
    
    return [best_start, best_len]
