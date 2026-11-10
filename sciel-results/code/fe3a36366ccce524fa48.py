"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Build a bitset of quadratic residues and store the first root for each residue.
    # root_for_residue[r] will store x such that x*x % p == r.
    # We initialize with 0 to indicate "not found" or "no root".
    root_for_residue = [0] * p
    residues = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if root_for_residue[r] == 0:
                root_for_residue[r] = x
            residues |= (1 << r)
    
    # Compute the bitset of starts s where s, s+1, s+2, s+3, s+4 are all QRs.
    # Bit s is set in runs iff bits s..s+4 are set in residues.
    runs = residues
    for i in range(1, 5):
        runs &= (residues >> i)
    
    # Find the smallest s >= 1 such that bit s is set in runs.
    # We can iterate or use bit tricks. Iteration is fine for p=1009.
    # To be faster, we can mask out bit 0 and find the lowest set bit.
    # But s must be >= 1.
    # Let's just iterate, it's very fast.
    for s in range(1, 1005):
        if (runs >> s) & 1:
            root_list = [root_for_residue[s + i] for i in range(5)]
            return {'start': s, 'roots': root_list}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
# EVOLVE-BLOCK-END
