import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # 1. Determine all non-zero quadratic residues and their smallest roots
    # residue -> smallest root r in [1, P-1]
    root_map = {}
    residue_mask = 0
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        # Since we iterate x from 1, the first occurrence is the smallest root
        if r not in root_map:
            root_map[r] = x
            residue_mask |= (1 << r)
            
    # 2. Scan for the smallest s
    # s must be in [1, 1004] so that s+4 <= 1008
    # We need s, s+1, s+2, s+3, s+4 to be set in residue_mask
    
    max_s = _P - 5
    # Pattern for 5 consecutive bits: 0b11111 = 31
    pattern = 31
    
    for s in range(1, max_s + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all 1 in residue_mask
        # (residue_mask >> s) & pattern == pattern
        if (residue_mask >> s) & pattern == pattern:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
