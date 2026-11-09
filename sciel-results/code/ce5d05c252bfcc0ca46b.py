"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    MOD = 1009
    N = 1008  # values are 1..1008
    
    # Compute modular inverse for each x in 1..1008
    # x * inv[x] ≡ 1 (mod 1009)
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    # Iterate over all possible starting positions L
    for L in range(1, N + 1):
        # Try to extend the interval as far as possible
        R = L
        
        # Check if single element [L, L] is valid
        # It is valid if inv[L] is not in [L, L], i.e., inv[L] != L
        # If inv[L] == L, length is 0 (or we skip, but problem says nonempty, 
        # however self-inverse elements 1 and 1008 make [1,1] and [1008,1008] invalid)
        if inv[L] == L:
            continue
            
        while R < N:
            next_val = R + 1
            inv_next = inv[next_val]
            
            # To extend to [L, R+1], we need:
            # 1. inv[R+1] not in [L, R+1]
            # 2. No x in [L, R] has inv[x] == R+1
            # Condition 2 is equivalent to inv[R+1] not in [L, R]
            # Combining 1 and 2: inv[R+1] not in [L, R+1]
            # This is true if inv[R+1] < L or inv[R+1] > R+1
            
            if inv_next < L or inv_next > next_val:
                R = next_val
            else:
                break
        
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
            
        # Deadline check
        if time.time() - start_time > DEADLINE:
            break
    
    # Fallback if no valid interval found (should not happen)
    if best_len == 0:
        best_start = 2
        best_len = 1
        
    return [best_start, best_len]

# EVOLVE-BLOCK-END
