def construct():
    p = 1009

    # Precompute: for each quadratic residue r^2 mod p (r in 1..p-1),
    # store one root. Use setdefault to keep the first (smallest) root found.
    root_of = {}
    for r in range(1, p):
        res = (r * r) % p
        root_of.setdefault(res, r)

    # Build a fast lookup: is_residue[i] for i in 0..p-1
    is_residue = [False] * p
    for res in root_of:
        if res != 0:  # zero residue excluded
            is_residue[res] = True

    # Find the smallest s in [1, 1004] such that s, s+1, ..., s+4 are all
    # non-zero quadratic residues mod 1009.
    for s in range(1, 1005):
        if (is_residue[s] and is_residue[s + 1] and is_residue[s + 2]
                and is_residue[s + 3] and is_residue[s + 4]):
            return {
                'start': s,
                'roots': [root_of[s], root_of[s + 1], root_of[s + 2],
                          root_of[s + 3], root_of[s + 4]]
            }

    # Fallback
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}
