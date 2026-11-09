import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    m = 5

    # Precompute roots: smallest x in [1, p-1] such that x^2 mod p == r
    root_map = {}
    for x in range(1, p):
        if time.time() > deadline:
            # Shouldn't happen, but safeguard
            break
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x

    # Set for O(1) residue membership check
    qr_set = set(root_map.keys())

    # Search for the minimum s in [1, p-m] such that s..s+m-1 are all nonzero QRs
    best_s = None
    best_roots = None

    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+m-1 are all in qr_set
        ok = True
        for i in range(m):
            if (s + i) not in qr_set:
                ok = False
                break
        if ok:
            best_s = s
            best_roots = [root_map[s + i] for i in range(m)]
            break  # first valid s is minimal

    if best_s is None:
        raise RuntimeError('No run of 5 consecutive nonzero quadratic residues found')

    return {'start': best_s, 'roots': best_roots}
# EVOLVE-BLOCK-END
