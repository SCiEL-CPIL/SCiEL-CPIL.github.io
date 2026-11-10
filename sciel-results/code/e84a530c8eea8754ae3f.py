import time

def construct():
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 ≡ 1 (mod 1009)
    # For prime p, x^2 ≡ 1 means x ≡ 1 or x ≡ p-1
    self_inv = {1, MOD - 1}
    
    best_start = 1
    best_length = 0
    
    # Iterate over all possible starts
    for s in range(1, N + 1):
        # Early termination: max possible length from s is N - s + 1
        if N - s + 1 <= best_length:
            break
            
        # Check if s itself is valid (not self-inverse)
        if s in self_inv:
            continue
        
        # Check time
        if time.time() > deadline:
            break
        
        # Try to extend from s
        for e in range(s, N + 1):
            if e in self_inv:
                break
                
            ie = inv[e]
            # If inv[e] is in [s, e], we have a conflict
            if s <= ie <= e:
                break
            
            current_length = e - s + 1
            if current_length > best_length:
                best_length = current_length
                best_start = s
                
    return [best_start, best_length]
