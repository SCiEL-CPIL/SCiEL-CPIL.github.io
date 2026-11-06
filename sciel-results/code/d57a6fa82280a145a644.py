"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 % MOD == 1 => x == 1 or x == MOD-1 = 1008
    self_inverse = {1, 1008}
    
    best_start = 1
    best_length = 0
    
    # Sliding window approach (inspiration 3):
    # For each right endpoint R, maintain the minimum valid left endpoint L.
    # When adding R to the window [L, R]:
    #   - If R is self-inverse, any window containing R is invalid, so L = R+1.
    #   - Otherwise, if inv[R] is in [L, R-1], we must push L past inv[R].
    #     This is the only new conflict when extending the window by R.
    # The window [L, R] is valid iff for all x in [L, R], inv[x] is not in [L, R].
    
    L = 1
    for R in range(1, N + 1):
        if time.time() > deadline:
            break
        
        if R in self_inverse:
            # Any window containing R is invalid
            L = R + 1
            continue
        
        # Check if inv[R] is in the current window [L, R-1]
        if inv[R] >= L and inv[R] < R:
            # inv[R] is in the current window, so we must exclude it
            L = inv[R] + 1
        
        # Also need to ensure L > R is not the case (window non-empty)
        if L <= R:
            current_length = R - L + 1
            if current_length > best_length:
                best_length = current_length
                best_start = L
    
    # If no valid interval was found (should not happen for N >= 2)
    if best_length == 0:
        for x in range(1, N + 1):
            if x not in self_inverse:
                best_start = x
                best_length = 1
                break
    
    # Verify the result for correctness
    s_final = best_start
    e_final = s_final + best_length - 1
    valid = True
    for x in range(s_final, e_final + 1):
        ix = inv[x]
        if s_final <= ix <= e_final:
            valid = False
            break
    
    if not valid:
        # Fallback: try a brute-force search for a valid interval
        # This should rarely be needed
        for s in range(1, N + 1):
            if time.time() > deadline:
                break
            in_window = [False] * (N + 1)
            for e in range(s, N + 1):
                if e in self_inverse:
                    break
                if in_window[inv[e]]:
                    break
                in_window[e] = True
                if e - s + 1 > best_length:
                    best_length = e - s + 1
                    best_start = s
                    valid = True
                else:
                    break
    
    if not valid:
        return [2, 30]
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
