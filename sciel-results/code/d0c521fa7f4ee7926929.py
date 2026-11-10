def construct():
    p = 1009
    # Precompute residues and their roots
    # is_res[r] is True if r is a non-zero quadratic residue mod p
    is_res = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_res[r] = True
            # Store the smallest root for determinism, though any valid root works
            if r not in root_map or x < root_map[r]:
                root_map[r] = x

    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all residues
    # s must be in [1, 1004] to ensure s+4 <= 1008 (no wraparound, no zero)
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all residues
        # Note: s+i are within 1..1008, so no modulo needed for indices
        if is_res[s] and is_res[s+1] and is_res[s+2] and is_res[s+3] and is_res[s+4]:
            roots = [root_map[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    # If no solution found (should not happen for p=1009, m=5)
    raise RuntimeError("No run of 5 quadratic residues found")
