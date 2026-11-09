import time

def construct():
    # Deadline guard: The task specifies 60s limit, but we add a safety margin.
    # For p=1009, this computation is instantaneous (< 0.1s), so this is just for compliance.
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    p = 1009
    roots = {}
    mask = 0
    
    # Precompute quadratic residues
    # x^2 mod p for x in 1..p-1
    # Note: 0 is excluded by the problem statement ("zero residues ... are excluded")
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots:
            roots[r] = x
        mask |= (1 << r)
        
    # Check deadline (unlikely to be hit)
    if time.time() - start_time > deadline:
        # Fallback: return a valid but potentially non-minimal result if we were still computing
        # For this specific instance, we are always done before this point.
        # We just need to return *something* valid to avoid timeout/exception.
        # However, since we are done, this block is effectively dead code for p=1009.
        pass

    # Find the lowest bit position s such that bits s..s+4 are all set in mask
    # This corresponds to s, s+1, s+2, s+3, s+4 all being non-zero quadratic residues.
    # consec has bit i set if mask has bits i, i+1, i+2, i+3, i+4 set.
    consec = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    
    # Get the lowest set bit position
    if consec == 0:
        # This should not happen for p=1009, but handle gracefully
        # Find any valid start? The problem guarantees a solution exists.
        # If no consecutive 5 residues, we can't form the required output.
        # But for p=1009, it exists.
        raise ValueError("No consecutive quadratic residues found")
        
    lsb = consec & (-consec)
    s = lsb.bit_length() - 1
    
    # Build the result
    result_roots = []
    for i in range(5):
        result_roots.append(roots[s + i])
        
    return {'start': s, 'roots': result_roots}
