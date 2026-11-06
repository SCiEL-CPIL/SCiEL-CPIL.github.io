import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 % MOD == 1 => x == 1 or x == MOD-1 = 1008
    self_inverse = set([1, 1008])
    
    best_start = 1
    best_length = 0
    
    L = 1
    for R in range(1, N + 1):
        # Check if R itself is self-inverse
        if R in self_inverse:
            # Any window containing R is invalid
            L = R + 1
            continue
        
        # Check if inv[R] is in [L, R-1] (i.e., already in the window)
        # If inv[R] >= L, we need to push L past inv[R]
        if inv[R] >= L and inv[R] < R:
            # inv[R] is in the current window [L, R-1], so we must exclude it
            # The new window must start after inv[R]
            L = max(L, inv[R] + 1)
        
        # Also need to check: for all x in [L, R-1], is inv[x] in [L, R]?
        # Actually, when we add R to the window, the only new conflicts are:
        # - inv[R] is in [L, R-1] (handled above)
        # - R is in [inv[x], ...] for some x in [L, R-1], which means inv[x] == R
        #   i.e., x == inv[R], which is the same condition
        # So we only need to check inv[R]
        
        # But wait, we also need to make sure that for all x in [L, R],
        # inv[x] is not in [L, R]. The previous window [L, R-1] was valid.
        # Adding R: the only new pair involving R is (R, inv[R]).
        # If inv[R] is in [L, R-1], we have a conflict.
        # If inv[R] == R, it's self-inverse (already handled).
        # If inv[R] > R, then inv[R] is not in [L, R] yet, so no conflict now.
        #   But later when we reach inv[R], we'll handle it.
        
        # So the only adjustment needed is: if inv[R] in [L, R-1], push L past inv[R]
        
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
    
    if best_length == 0:
        # Fallback: find any valid single element
        # 2 is valid since inv[2] = 505, and 2 != 505, and 2 is not self-inverse
        return [2, 1]
    
    return [best_start, best_length]
