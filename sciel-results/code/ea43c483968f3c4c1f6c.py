import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_start = 1
    best_length = 0
    
    # Sweep over all possible start positions L
    # Note: L=1 and L=1008 are self-inverse, so they can't form valid intervals of length > 0
    # Actually, if L=1, the interval contains 1, and 1*1=1, so it's invalid.
    # Similarly for L=1008.
    # So we can start from L=2 to 1007 for potential improvements, but let's keep the general loop
    # and let the logic handle it. The logic breaks immediately for L=1 because inv[1]=1 is in [1,1].
    
    for L in range(1, N + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        # Expand R from L upwards
        R = L - 1
        while R < N:
            y = R + 1
            # The new element is y.
            # The interval would be [L, y].
            # It is valid iff inv[y] is NOT in [L, y].
            if L <= inv[y] <= y:
                break
            R = y
            
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
            
    return [best_start, best_length]
