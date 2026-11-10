# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Build a bitset of quadratic residues modulo p (excluding 0).
    # bit i is set if i is a nonzero quadratic residue mod p.
    residues = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            residues |= (1 << r)
    
    # Compute a bitset 'runs' where bit s is set iff s, s+1, s+2, s+3, s+4 are all QRs.
    # This is done by AND-ing the residues bitset with its shifted versions.
    runs = residues
    for i in range(1, 5):
        runs &= (residues >> i)
        
    # Find the smallest s >= 1 such that bit s is set in 'runs'.
    # Since bit 0 of 'residues' is 0, bit 0 of 'runs' is 0.
    # Thus, the lowest set bit in 'runs' corresponds to the minimum valid s.
    if runs == 0:
        raise RuntimeError('No run of 5 consecutive quadratic residues found')
        
    # Isolate the lowest set bit and find its index.
    lowest_bit = runs & -runs
    s = lowest_bit.bit_length() - 1
    
    # Verify s is within bounds [1, 1004]
    if s < 1 or s > 1004:
        raise RuntimeError('Start index out of valid range')
        
    # Pre-compute a lookup table for roots to avoid dictionary overhead.
    # root_table[r] stores a valid root x for residue r.
    root_table = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if root_table[r] == 0:
            root_table[r] = x
            
    # Construct the list of roots for s, s+1, ..., s+4
    root_list = [root_table[s + i] for i in range(5)]
    
    return {'start': s, 'roots': root_list}
# EVOLVE-BLOCK-END
