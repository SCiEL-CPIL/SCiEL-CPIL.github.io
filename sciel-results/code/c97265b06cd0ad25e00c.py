import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    p = 1009
    m = 5
    
    # Precompute quadratic residues and one root for each
    # roots[r] = x such that x*x % p == r
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
            
    qr_set = set(roots.keys())
    
    # Search for the minimum s
    # s ranges from 1 to p - m
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            # This should not happen given the small p, but required by directive
            # Fallback: return a valid late run if we had one, or raise
            # Since we are iterating from 1, if we haven't found one, 
            # we might need a fallback. But we are almost certain to find one early.
            # For robustness, if we hit deadline, we can try a random late s.
            # But let's assume we find it.
            pass
        
        # Check if s, s+1, ..., s+4 are all quadratic residues
        # Optimization: check one by one to break early
        is_run = True
        for i in range(m):
            if (s + i) not in qr_set:
                is_run = False
                break
        
        if is_run:
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
            
    # If no run found (highly unlikely), raise error
    raise RuntimeError('No run found')
# EVOLVE-BLOCK-END
