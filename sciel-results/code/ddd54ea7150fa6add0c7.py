"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses for 1 to N
    # inv[x] is the unique y in 1..N such that x*y % MOD == 1
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # For each element x, define its "conflict interval" as [min(x, inv[x]), max(x, inv[x])]
    # An interval [s, e] is valid iff for every x in [s, e], the conflict interval of x
    # does not contain any other element of [s, e] (except x itself).
    # Equivalently: for all x in [s, e], inv[x] < s or inv[x] > e.
    
    # We use a two-pointer / sliding window approach:
    # For a fixed start s, find the maximum end e such that [s, e] is valid.
    # As s increases, e is non-decreasing (monotonicity property).
    
    best_start = 1
    best_length = 0
    e = 0  # current right boundary
    
    for s in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # Ensure e >= s
        if e < s:
            e = s
            
        # Check if the start element s itself is a self-inverse.
        # If inv[s] == s, then [s, s] is invalid, and any interval containing s is invalid.
        # Since we are fixing the start at s, no valid interval starts at s.
        if inv[s] == s:
            continue
            
        # Expand e as far as possible while [s, e] remains valid
        while e < N:
            next_e = e + 1
            # Condition for [s, next_e] to be valid given [s, e] is valid:
            # inv[next_e] must not be in [s, next_e].
            # This is equivalent to inv[next_e] < s or inv[next_e] > next_e.
            if inv[next_e] < s or inv[next_e] > next_e:
                e = next_e
            else:
                break
                
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
            
        # Since e is non-decreasing as s increases, we can keep e as is for the next s.
        # But we need to ensure e >= s for the next iteration.
        
    return [best_start, best_length]
# EVOLVE-BLOCK-END
