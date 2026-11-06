"""Optimized program using bitwise convolution approach for finding earliest run of 5 consecutive quadratic residues."""

def construct():
    p = 1009
    # Precompute roots and build bitset in a single pass
    # residue_to_root[r] = x where x^2 % p == r
    residue_to_root = {}
    bitset = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We only need one root per residue. The first one encountered is fine.
            # Since we iterate x from 1, we get the smallest positive root.
            if r not in residue_to_root:
                residue_to_root[r] = x
            bitset |= (1 << r)
    
    # Find smallest s (1 <= s <= 1004) such that s, s+1, s+2, s+3, s+4 are all residues
    # Check: (bitset >> s) & 0b11111 == 0b11111
    mask = 0b11111  # 5 consecutive bits
    for s in range(1, 1005):
        if (bitset >> s) & mask == mask:
            # Found the start, retrieve precomputed roots
            roots = []
            for i in range(5):
                val = s + i
                roots.append(residue_to_root[val])
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
