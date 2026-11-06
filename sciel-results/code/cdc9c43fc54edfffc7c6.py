def construct():
    P = 1009
    M = 5
    
    # Bitset where bit i is 1 if i is a non-zero quadratic residue mod P
    # We use a Python integer as an arbitrary-precision bitset
    residue_bitset = 0
    # Map from residue to one valid root (1 <= root <= P-1)
    root_map = {}
    
    # Precompute quadratic residues
    for x in range(1, P):
        r = (x * x) % P
        if r != 0:
            # Set the bit at position r
            residue_bitset |= (1 << r)
            # Store the first root found for this residue
            if r not in root_map:
                root_map[r] = x
                
    # We need to find the minimum s in [1, P-M] such that
    # s, s+1, ..., s+M-1 are all quadratic residues.
    # Since s <= 1004 and M=5, max index is 1008, which is < P=1009.
    # So we don't need to worry about wrap-around in the index check itself,
    # but the bitset check naturally handles the range.
    
    # Mask for M consecutive bits: 0b11111 = 31
    mask = (1 << M) - 1
    
    for s in range(1, P - M + 1):
        # Shift the bitset right by s, so bit 0 corresponds to index s.
        # Then check if the lowest M bits are all 1.
        if ((residue_bitset >> s) & mask) == mask:
            # Found the minimal s
            roots = [root_map[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No valid run found')
