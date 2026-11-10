"""Rewritten program to find the minimum start s for a run of 5 quadratic residues."""

def construct():
    p = 1009
    # Map residue to a root
    # We need roots for residues 1..1008
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the first root found (smallest x)
            if r not in roots:
                roots[r] = x
    
    # Search for the minimum s
    # s must be such that s..s+4 are all in [1, 1008]
    # So s <= 1008 - 4 = 1004
    # Also s >= 1
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # They must be in the keys of 'roots'
        if all((s + i) in roots for i in range(5)):
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
    
    # If no run found, which shouldn't happen for p=1009
    raise RuntimeError("No run of 5 quadratic residues found")
