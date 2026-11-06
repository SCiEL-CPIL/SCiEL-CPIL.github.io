import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    P = 1009
    
    # Precompute quadratic residues and their roots
    # roots[r] = x such that x*x % P == r
    # We only care about non-zero residues
    roots = {}
    is_qr = [False] * P
    
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            is_qr[r] = True
            # Store the first root found for each residue
            if r not in roots:
                roots[r] = x
    
    # Search for the minimum s
    # s must be such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # Given 1 <= s <= 1004, s+4 <= 1008, so no modulo wraparound for the indices themselves
    for s in range(1, 1005):
        if time.time() > deadline:
            break
            
        # Check if s, s+1, ..., s+4 are all in QR
        valid = True
        for i in range(5):
            val = s + i
            if not is_qr[val]:
                valid = False
                break
        
        if valid:
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
            
    # Fallback if not found (should not happen for this prime)
    raise RuntimeError("No run found")
