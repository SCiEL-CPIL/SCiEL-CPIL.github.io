"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute roots for all non-zero quadratic residues.
    # We need ri such that ri**2 mod p == s+i, with 1 <= ri <= p-1.
    # For each residue value r (1 to p-1), find a root x in [1, p-1] such that x*x % p == r.
    # We'll store the smallest such root for efficiency, though any valid root works.
    
    roots = {}
    for x in range(1, p):
        r = x * x % p
        if r != 0:  # Exclude zero residues
            if r not in roots:
                roots[r] = x
    
    # Now find the minimum s such that s, s+1, s+2, s+3, s+4 are all in roots.
    # s ranges from 1 to p-5 = 1004 (since s+4 <= 1008 = p-1, so s <= p-5)
    
    # Use a bitset approach for efficiency:
    # Create a bitset where bit i is set if residue i is a quadratic residue (non-zero).
    # Then find the smallest s where bits s through s+4 are all set.
    
    # Build the bitset
    bitset = 0
    for r in roots:
        bitset |= (1 << r)
    
    # We need to find the smallest s in [1, p-5] such that:
    # (bitset >> s) & 0b11111 == 0b11111
    # i.e., bits s, s+1, s+2, s+3, s+4 are all set.
    
    # To find the minimum s efficiently:
    # Create a shifted version and AND them together.
    # mask = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    # Then the set bits in mask correspond to valid starting positions s.
    # We want the smallest s >= 1 that is set in mask.
    
    mask = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Find the least significant set bit in mask that is >= 1.
    # Clear bit 0 if it's set (since s must be >= 1).
    mask &= ~1  # Clear bit 0
    
    # Find the position of the least significant set bit.
    if mask == 0:
        raise RuntimeError('No valid run found')
    
    # Get the index of the lowest set bit
    s = (mask & -mask).bit_length() - 1
    
    # Verify s is in valid range
    if s < 1 or s > p - 5:
        # Fallback: linear scan
        for s in range(1, p - 4):
            if all((s + i) in roots for i in range(5)):
                break
        else:
            raise RuntimeError('No valid run found')
    
    # Construct the roots list
    result_roots = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': result_roots}
