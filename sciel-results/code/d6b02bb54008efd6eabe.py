def construct():
    p = 1009
    residues = 0
    roots = {}
    
    # Single pass to build bitset and roots map
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residues |= (1 << r)
            if r not in roots:
                roots[r] = x
                
    # Find runs of 5 consecutive QRs
    # runs bit s is 1 if s, s+1, s+2, s+3, s+4 are all QRs
    runs = residues
    for i in range(1, 5):
        runs &= (residues >> i)
        
    # We need s >= 1. Bit 0 corresponds to s=0. Clear it.
    runs &= ~1
    
    if runs == 0:
        raise RuntimeError('No run found')
        
    # Find the smallest s (lowest set bit)
    lsb = runs & -runs
    s = lsb.bit_length() - 1
    
    # Verify s is within bounds (should be, given the problem setup)
    if s < 1 or s > 1004:
        raise RuntimeError(f'Invalid start {s}')
        
    root_list = [roots[s + i] for i in range(5)]
    return {'start': s, 'roots': root_list}
