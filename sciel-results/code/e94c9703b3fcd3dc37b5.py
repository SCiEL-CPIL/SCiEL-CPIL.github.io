import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0

def construct():
    # 1. Determine all non-zero quadratic residues and their smallest roots
    root_map = {}
    is_residue = [False] * _P
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x
            is_residue[r] = True
            
    # 2. Scan for the smallest s
    # s must be in [1, 1004] so that s+4 <= 1008
    max_s = _P - 5
    
    for s in range(1, max_s + 1):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if (is_residue[s] and is_residue[s+1] and is_residue[s+2] and 
            is_residue[s+3] and is_residue[s+4]):
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
