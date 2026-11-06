"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute quadratic residues and one root for each
    # x ranges from 1 to p-1
    # r = x^2 % p
    # We want the minimum s, so we just need to know if a number is a residue
    # and have a valid root for it.
    
    is_residue = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # r will be in 1..p-1 because x != 0 mod p
        if not is_residue[r]:
            is_residue[r] = True
            roots[r] = x
    
    # Search for the minimum start s
    # s must be such that s, s+1, s+2, s+3, s+4 are all residues
    # s ranges from 1 to 1004 (so s+4 <= 1008)
    for s in range(1, p - 4): # 1 to 1004
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        if (is_residue[s] and is_residue[s+1] and is_residue[s+2] and 
            is_residue[s+3] and is_residue[s+4]):
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
            
    # This should never happen for p=1009
    raise RuntimeError("No run of 5 quadratic residues found")
# EVOLVE-BLOCK-END
