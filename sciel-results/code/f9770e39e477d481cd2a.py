"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of non-zero quadratic residues modulo p.
    # bit i is 1 iff i is a non-zero QR mod p.
    # We only care about residues 1..p-1 (i.e., 1..1008).
    bitset = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bitset |= (1 << r)
    
    # We want the smallest s in [1, 1004] such that s, s+1, s+2, s+3, s+4 are all non-zero QRs.
    # That is, bits s, s+1, s+2, s+3, s+4 are all set in bitset.
    #
    # Using bit operations: for a given s, the condition is:
    #   (bitset >> s) & 0b11111 == 0b11111
    #
    # To find the minimum s efficiently, we can iterate s from 1 upward and check this condition.
    # But we can do better: precompute a "window" mask approach.
    #
    # Alternative: compute a new bitset 'consec' where bit s is 1 iff bits s..s+4 are all set.
    # consec = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    # Then find the lowest set bit in consec (within range 1..1004).
    
    consec = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    # Now find the smallest s in [1, 1004] such that bit s of consec is 1.
    # We scan from s=1 upward.
    for s in range(1, 1005):
        if (consec >> s) & 1:
            # Found the start. Now we need to find a root for each s+i.
            # Build a lookup: for each residue r (1..1008), store one root x in [1, 1008] such that x^2 mod p == r.
            # We can reuse the loop or precompute.
            roots_lookup = {}
            for x in range(1, p):
                r = (x * x) % p
                if r != 0 and r not in roots_lookup:
                    roots_lookup[r] = x
            roots = [roots_lookup[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of five consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
