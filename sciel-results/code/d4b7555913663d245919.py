"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    N = 1008

    # Precompute modular inverses using Fermat's little theorem
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    # Self-inverse elements: x where x*x % p == 1
    # For prime p=1009, these are x=1 and x=p-1=1008
    self_inv = {1, 1008}

    best_start = 1
    best_len = 0

    # Sliding window / Two-pointer approach
    # s is the start of the current valid window [s, e]
    s = 1
    
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        if e in self_inv:
            # Cannot include e in any valid interval
            # Move start past e
            s = e + 1
        else:
            ie = inv[e]
            # Check if ie is in the current window [s, e-1]
            # If s <= ie <= e-1, there is a conflict
            if s <= ie <= e - 1:
                # Must move s to be after ie
                s = ie + 1
            
            # Now [s, e] is valid
            cur_len = e - s + 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = s

    return [best_start, best_len]
# EVOLVE-BLOCK-END
