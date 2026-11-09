def construct():
    p = 1009
    # Precompute quadratic residues and one root for each
    roots = {}
    for x in range(1, p):
        v = (x * x) % p
        if v not in roots:
            roots[v] = x
    # Search for the minimum s such that s, s+1, ..., s+4 are all quadratic residues
    # No wraparound: s+4 <= 1008, so s <= 1004
    for s in range(1, 1005):
        if all((s + i) in roots for i in range(5)):
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
    raise RuntimeError('No run found')
