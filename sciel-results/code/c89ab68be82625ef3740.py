import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # 1. Precompute quadratic residues and their smallest roots
    # is_res[i] is True if i is a non-zero quadratic residue mod _P
    is_res = [False] * (_P)
    # root_map[r] stores the smallest root x in [1, _P-1] such that x^2 % _P == r
    root_map = {}
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        if not is_res[r]:
            is_res[r] = True
            root_map[r] = x
        # Note: Since we iterate x from 1 upwards, the first time we see a residue r,
        # x is the smallest root. We don't need to update root_map if is_res[r] is already True.

    # 2. Sliding window to find the smallest s
    # We look for s in [1, _P-5] such that s, s+1, s+2, s+3, s+4 are all residues.
    
    # Initialize window count for s=1: residues in [1, 5]
    window_count = 0
    for i in range(1, 6):
        if is_res[i]:
            window_count += 1
            
    max_s = _P - 5
    for s in range(1, max_s + 1):
        # Check if current window [s, s+4] has 5 residues
        if window_count == 5:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
        
        # Update window for next s (if any)
        if s < max_s:
            # Remove s from window
            if is_res[s]:
                window_count -= 1
            # Add s+5 to window
            if is_res[s + 5]:
                window_count += 1

    # If no run found (should not happen for p=1009), raise error
    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
