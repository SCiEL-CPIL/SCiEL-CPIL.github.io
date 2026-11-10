"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    MOD = 1009
    
    # Precompute modular inverses for 1 to 1008
    # inv[x] = x^(-1) mod 1009
    inv = [0] * MOD
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    best_start = 2
    best_length = 0
    
    # The interval must be within [2, 1007] because 1 and 1008 are self-inverses
    # (1*1=1, 1008*1008 = 1 mod 1009)
    
    for s in range(2, 1008):
        if time.time() > deadline:
            break
            
        e = s
        # Extend e as far as possible
        # Condition for [s, e] to be valid: for all x in [s, e], inv[x] is NOT in [s, e]
        # When adding new element e, we need inv[e] not in [s, e]
        # Also, we need to ensure that no previous x in [s, e-1] has inv[x] == e.
        # This is equivalent to inv[e] not in [s, e-1].
        # So the combined condition is: inv[e] not in [s, e].
        
        while e < 1007:
            if time.time() > deadline:
                break
            new_e = e + 1
            # Check if adding new_e keeps the interval valid
            # We need inv[new_e] to be outside [s, new_e]
            if inv[new_e] < s or inv[new_e] > new_e:
                e = new_e
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
