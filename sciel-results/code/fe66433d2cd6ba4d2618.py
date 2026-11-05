import time


def construct():
    """
    Find the minimum s in [1, 1004] such that s, s+1, s+2, s+3, s+4
    are all non-zero quadratic residues modulo 1009.

    Returns a dict with 'start' (the minimum s) and 'roots' (list of 5 integers,
    one per residue, such that root**2 % 1009 equals the residue).
    """
    p = 1009
    m = 5

    # Precompute roots: root_map[r] = smallest x in [1, p-1] with x*x % p == r,
    # or 0 if r is not a non-zero quadratic residue.
    root_map = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_map[r] == 0:
            root_map[r] = x

    # Build a boolean array is_qr where is_qr[v] is True iff v is a non-zero QR mod p.
    is_qr = [False] * (p + 5)  # a little extra padding to avoid index issues
    for v in range(1, p):
        if root_map[v] != 0:
            is_qr[v] = True

    # Precompute a "run length" array: run_len[i] = length of consecutive QRs starting at i.
    # We compute it from the end backwards.
    run_len = [0] * (p + 5)
    for i in range(p - 1, 0, -1):
        if is_qr[i]:
            run_len[i] = run_len[i + 1] + 1
        else:
            run_len[i] = 0

    # Find the smallest s in [1, p-m] with run_len[s] >= m
    for s in range(1, p - m + 1):
        if run_len[s] >= m:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback (should not happen for p=1009)
    raise RuntimeError("No valid run of 5 consecutive non-zero QRs found modulo 1009")
