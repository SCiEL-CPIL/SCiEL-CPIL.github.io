"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # bit i is 1 iff i is a non-zero QR mod p
    bitset = 0
    # roots_list[r] = x such that x^2 % p == r, for r in 1..p-1
    # Initialize with 0 (which is not a valid root index for non-zero residues in our range 1..1008)
    roots_list = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            bitset |= (1 << r)
            # Store the first root found for each residue
            if roots_list[r] == 0:
                roots_list[r] = x
                
    # Find the smallest s in [1, 1004] such that s..s+4 are all QRs
    # consec bit s is 1 iff bits s, s+1, s+2, s+3, s+4 are all set in bitset
    consec = bitset & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    
    for s in range(1, 1005):
        if (consec >> s) & 1:
            roots = [roots_list[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    raise RuntimeError('No run of five consecutive non-zero quadratic residues found')
# EVOLVE-BLOCK-END
