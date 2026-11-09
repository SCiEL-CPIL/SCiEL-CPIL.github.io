def construct():
    p = 1009
    # Precompute the minimal root for each quadratic residue in [1, p-1]
    # Using a list indexed by the residue value for O(1) access
    root_of = [0] * p
    is_res = [False] * p

    for x in range(1, p):
        v = (x * x) % p
        if v == 0:
            continue
        if not is_res[v]:
            is_res[v] = True
            root_of[v] = x
        # Note: we keep the first root found, which is the smaller root
        # since we iterate x from 1 upward

    # Search for the smallest s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero)
    # and s+4 <= 1008 (to avoid wraparound and stay in valid range)
    for s in range(1, 1005):
        if s + 4 > 1008:
            break
        ok = True
        for i in range(5):
            val = s + i
            if not is_res[val]:
                ok = False
                break
        if ok:
            return {'start': s, 'roots': [root_of[s + i] for i in range(5)]}

    raise RuntimeError("No run of 5 consecutive quadratic residues found")
