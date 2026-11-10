import time

def construct():
    start_time = time.time()
    deadline = 80.0
    
    p = 1009
    
    # is_qr[i] is True if i is a quadratic residue mod p
    is_qr = [False] * p
    # root_of[i] stores a root r such that r*r % p == i
    root_of = [-1] * p
    
    # 0 is a residue (0^2), but problem excludes zero residues from the run
    # We only care about 1..1008
    for r in range(1, p):
        res = (r * r) % p
        if not is_qr[res]:
            is_qr[res] = True
            root_of[res] = r
            
    # Find the smallest s
    # s must be in [1, 1004] so that s+4 <= 1008
    for s in range(1, 1005):
        if time.time() - start_time > deadline:
            break
            
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        # We can short-circuit
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {
                'start': s,
                'roots': [root_of[s], root_of[s+1], root_of[s+2], root_of[s+3], root_of[s+4]]
            }
            
    # Fallback if not found (should not happen for p=1009)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
