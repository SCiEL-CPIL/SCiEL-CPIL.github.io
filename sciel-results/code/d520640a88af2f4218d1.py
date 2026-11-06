"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s wall-clock limit
    
    # Precompute quadratic residues and their roots
    # For each residue r in [1, 1008], store a root x such that x^2 % 1009 == r
    roots = {}
    for x in range(1, 1009):
        r = (x * x) % 1009
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Search for the minimum s such that s, s+1, s+2, s+3, s+4 are all quadratic residues
    # s must be in [1, 1004] and s+4 <= 1008, so s <= 1004
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        valid = True
        res_roots = []
        for i in range(5):
            val = s + i
            if val > 1008 or val not in roots:
                valid = False
                break
            res_roots.append(roots[val])
        if valid:
            return {'start': s, 'roots': res_roots}
    
    # Fallback: if for some reason we didn't find it in ascending order (shouldn't happen),
    # try descending to ensure we return something valid
    for s in range(1004, 0, -1):
        if time.time() > deadline:
            break
        valid = True
        res_roots = []
        for i in range(5):
            val = s + i
            if val > 1008 or val not in roots:
                valid = False
                break
            res_roots.append(roots[val])
        if valid:
            return {'start': s, 'roots': res_roots}
    
    raise RuntimeError('No baseline run found')
# EVOLVE-BLOCK-END
