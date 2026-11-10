"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find the longest nonempty interval [start, start+length-1] contained in [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    P = 1009
    MAX_VAL = 1008  # residues 1..1008
    
    # Precompute modular inverses
    inv = [0] * (P + 1)
    for x in range(1, P):
        inv[x] = pow(x, P - 2, P)
    
    # Convert to numpy array for vectorized operations
    inv_arr = np.array(inv, dtype=np.int32)
    
    # Self-inverse residues: x^2 ≡ 1 mod 1009 → x = 1 or x = 1008
    # These cannot be in any valid interval
    
    best_start = 2
    best_length = 0
    
    # Greedy extension for each start position
    # For a fixed start s, extend e as far as possible.
    # The interval [s, e] is valid iff for all x in [s, e], inv[x] is not in [s, e].
    # When extending from e to e+1, we need inv[e+1] not in [s, e+1].
    # Since inv is a bijection, inv[e+1] == some value v.
    # We need v < s or v > e+1.
    
    for s in range(2, 1008):  # s from 2 to 1007 (1 and 1008 are self-inverse)
        if time.time() > deadline:
            break
        
        e = s
        # Extend e as far as possible
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
    
    # Verify the result
    if best_length > 0:
        s = best_start
        e = s + best_length - 1
        valid = True
        for x in range(s, e + 1):
            if x * x % P == 1:  # self-inverse check
                valid = False
                break
            y = inv[x]
            if s <= y <= e:
                valid = False
                break
        if valid:
            return [s, best_length]
    
    # Fallback: try a simple valid interval
    return [2, 30]
# EVOLVE-BLOCK-END
