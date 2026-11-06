"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    MOD = 1009
    
    # Precompute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 2
    best_len = 1
    
    # Local variables for speed
    inv_local = inv
    mod_local = MOD
    deadline_local = deadline
    
    for s in range(2, mod_local - 1):
        if time.time() > deadline_local:
            break
        
        max_end = s
        # We scan e from s upwards.
        # We must stop if:
        # 1. e >= 1008
        # 2. inv[e] is in [s, e-1] (conflict with past)
        # 3. We have previously established an upper bound from a forward inverse that is now violated.
        #    Note: The current code relies on hitting the forward inverse to break.
        #    This is correct.
        
        for e in range(s, mod_local):
            if time.time() > deadline_local:
                break
            if e >= 1008:
                break
            
            ie = inv_local[e]
            
            if ie == e:
                break
            elif ie < s:
                # Inverse is before the start, so no conflict.
                max_end = e
            elif ie < e:
                # Inverse is in [s, e-1], so conflict!
                break
            else:
                # Inverse is ahead (ie > e).
                # We can include e, but this implies we must stop before ie.
                # The loop will naturally break when it reaches ie (if ie < mod_local).
                # If ie >= mod_local, it's out of bounds, so no conflict within [1, 1008].
                max_end = e
        
        L = max_end - s + 1
        if L > best_len:
            best_len = L
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
