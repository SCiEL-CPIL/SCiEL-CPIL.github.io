import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    L = 1
    active = set()
    max_len = 0
    best_start = 1
    
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
            
        if inv[R] == R:
            # Self-inverse, cannot be in interval
            # Reset window
            L = R + 1
            active.clear()
        else:
            active.add(R)
            # Resolve conflicts
            # If inv[R] is in active, we must move L past it
            while inv[R] in active:
                active.remove(L)
                L += 1
                
            current_len = R - L + 1
            if current_len > max_len:
                max_len = current_len
                best_start = L
                
    return [best_start, max_len]
