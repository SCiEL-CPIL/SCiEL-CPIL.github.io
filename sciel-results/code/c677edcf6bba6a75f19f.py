import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    p = 1009
    m = 5

    # Precompute roots: smallest x in [1, p-1] such that x^2 mod p == r
    # Using a direct array indexed by residue value for O(1) lookup
    # root_arr[r] = smallest x such that x^2 mod p == r, or 0 if r is not a QR
    root_arr = [0] * p
    is_residue = [False] * p

    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r == 0:
            continue
        if root_arr[r] == 0:  # first (smallest) root for this residue
            root_arr[r] = x
            is_residue[r] = True

    # Search for the minimum s in [1, p-m] such that s..s+m-1 are all nonzero QRs
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
        # Check if s, s+1, ..., s+m-1 are all nonzero quadratic residues
        ok = True
        for i in range(m):
            if not is_residue[s + i]:
                ok = False
                break
        if ok:
            roots = [root_arr[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive nonzero quadratic residues found')
