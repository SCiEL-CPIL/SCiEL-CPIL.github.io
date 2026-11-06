import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    
    R = 0
    for L in range(1, MOD):
        if time.time() > deadline:
            break
            
        # Ensure R is at least L
        if R < L:
            R = L
            
        # If L is self-inverse, it cannot be in any valid interval.
        # So we must move R at least to L?
        # If L is self-inverse, any interval starting at L is invalid.
        # So we should effectively skip L as a start?
        # Or set R such that L is not included?
        # But the interval must start at L?
        # "Return [start, length] for a nonempty interval contained in [1, 1008]."
        # "For EVERY x,y in the interval... require x*y %1009 !=1."
        # If L is self-inverse, L*L % 1009 == 1. So L cannot be in the interval.
        # So no valid interval starts at L and includes L.
        # Thus, the max length for start L is 0.
        # We can just continue.
        if L == inv[L]:
            continue

        # Expand R as far as possible
        # We need to check if adding R+1 is valid.
        # Condition for R+1 to be added to [L, R]:
        # 1. R+1 <= N
        # 2. inv[R+1] is NOT in [L, R+1]
        
        while R < N:
            next_R = R + 1
            # Check if next_R conflicts with [L, next_R]
            if L <= inv[next_R] <= next_R:
                break
            R = next_R
            
        # Current valid interval is [L, R]
        # But we must ensure L is not self-inverse (handled above)
        # And that [L, R] is valid.
        # The while loop ensures that R is the max such that [L, R] is valid?
        # Let's trace.
        # Initially R might be < L.
        # If R < L, we set R = L.
        # But if L is self-inverse, we continue.
        # If L is not self-inverse, is [L, L] valid?
        # Yes, because L != inv[L].
        # So R should be at least L.
        
        # Wait, if R was expanded in a previous iteration of L, it might be > L.
        # Is [L, R] valid?
        # When L was L-1, [L-1, R] was valid.
        # Removing L-1 from the left cannot create new conflicts.
        # So [L, R] is valid.
        
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L

    return [best_start, best_len]
