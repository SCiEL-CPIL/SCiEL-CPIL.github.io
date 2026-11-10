def construct():
    p = 1009
    # Precompute a mapping from quadratic residue to one of its roots.
    # We iterate x from 1 to p-1.
    # For each x, r = x*x % p.
    # If r is not 0, we can store r -> x.
    # We don't need to store the smallest root, just any valid root.
    # The problem statement says "Each residue is certified by its root", implying any root is fine.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We just need one root. The first one found is fine.
            # Note: If we want deterministic output, this loop is deterministic.
            if r not in roots:
                roots[r] = x
    
    # We need to find the minimum s such that s, s+1, s+2, s+3, s+4 are all in roots.
    # Constraints: 1 <= s <= 1004, so that s+4 <= 1008 (non-zero residues).
    # Search s in ascending order to minimize s.
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if all((s + i) in roots for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
    
    # This should not happen for p=1009, but handle gracefully if it does.
    raise RuntimeError("No valid start found")
