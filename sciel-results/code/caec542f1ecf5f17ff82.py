"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MOD = 1009
    MAX_VAL = MOD - 1  # 1008

    # Precompute modular inverses for 1 to 1008
    # inv[x] = x^(-1) mod 1009
    inv = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_length = 0
    
    # The interval must be within [2, 1007] because 1 and 1008 are self-inverse (1*1=1, 1008*1008=1 mod 1009)
    # So any interval containing 1 or 1008 is invalid.
    # Thus s >= 2 and e <= 1007.
    
    # For each start s, find the maximum valid end e.
    # An interval [s, e] is valid iff for all x in [s, e], inv[x] is NOT in [s, e].
    # This is equivalent to: no x in [s, e] has inv[x] in [s, e].
    
    # Efficient approach: for each s, we can extend e as far as possible.
    # When we add a new element next_e, we need inv[next_e] not in [s, next_e].
    # Also, we need to ensure that for all existing elements x in [s, next_e-1], 
    # inv[x] is not in [s, next_e]. But since inv[x] was already checked when x was added,
    # and inv[x] < next_e would have been caught... wait, no.
    
    # Actually, the condition is: for all x in [s, e], inv[x] not in [s, e].
    # When extending from [s, e] to [s, e+1], we need:
    # 1. inv[e+1] not in [s, e+1]
    # 2. For all x in [s, e], inv[x] not in [s, e+1]
    #    But condition 2 is automatically satisfied if it was satisfied for [s, e],
    #    UNLESS inv[x] == e+1 for some x in [s, e].
    #    So we also need: for all x in [s, e], inv[x] != e+1.
    #    This is equivalent to: e+1's inverse is not in [s, e], which is the same as condition 1!
    #    Wait, no. inv[x] == e+1 means x == inv[e+1]. So we need inv[e+1] not in [s, e].
    #    Combined with condition 1 (inv[e+1] != e+1, which is always true for e+1 not in {1, 1008}),
    #    the total condition is: inv[e+1] not in [s, e+1], which simplifies to inv[e+1] not in [s, e] 
    #    since inv[e+1] != e+1 for e+1 in [2, 1007].
    
    # So the greedy extension is correct: just check if inv[next_e] is in [s, next_e].
    
    for s in range(2, MAX_VAL):  # s from 2 to 1007
        if time.time() > deadline:
            break
            
        e = s
        
        # Try to extend the interval to the right
        while e < MAX_VAL - 1:  # e can go up to 1007
            next_e = e + 1
            # Check if adding next_e keeps the interval valid.
            # Condition: inv[next_e] must NOT be in [s, next_e]
            if not (s <= inv[next_e] <= next_e):
                e = next_e
            else:
                break
        
        length = e - s + 1
        if length > best_length:
            best_length = length
            best_start = s

    # Fallback if no valid interval found (should not happen as [2,2] is valid)
    if best_length == 0:
        best_start = 2
        best_length = 1

    return [best_start, best_length]
# EVOLVE-BLOCK-END
