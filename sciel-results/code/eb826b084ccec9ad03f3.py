"""Find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute roots: map each non-zero quadratic residue to a root x in [1, p-1]
    # such that x*x % p == residue.
    # We only need one root per residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Build a boolean array indicating which values in [1, p-1] are non-zero quadratic residues
    is_residue = [False] * (p)  # indices 0..p-1
    for r in roots:
        is_residue[r] = True
    
    # Search for the smallest s in [1, p-5] such that s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues
    # Note: s+i must be in [1, p-1], so s <= p-5 = 1004
    for s in range(1, p - 4):  # s from 1 to 1004
        if all(is_residue[s + i] for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
