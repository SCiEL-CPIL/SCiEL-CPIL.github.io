import time

P = 1009
M = 5


def _find_runs(is_qr):
    """Return the smallest s in [1, P-M] such that is_qr[s..s+M-1] are all True.
    Pure-Python sliding window check.
    """
    n = len(is_qr)
    # is_qr is 0-indexed: index 0..P-1 correspond to residues 0..P-1
    # We need 1-indexed s such that residues s, s+1, ..., s+M-1 are all QR
    # i.e., is_qr[s], is_qr[s+1], ..., is_qr[s+M-1] are all True (0-indexed)
    for s in range(1, P - M + 1):
        ok = True
        for i in range(M):
            if not is_qr[s + i]:
                ok = False
                break
        if ok:
            return s
    return -1


def construct():
    t0 = time.time()

    # Precompute QR table and one root per residue.
    is_qr = [False] * P
    roots = [0] * P

    for xv in range(1, P):
        rv = (xv * xv) % P
        if rv != 0:
            is_qr[rv] = True
            if roots[rv] == 0:
                roots[rv] = xv

    s = _find_runs(is_qr)
    if s == -1:
        raise RuntimeError("No run of 5 consecutive non-zero QRs found mod 1009")

    result_roots = [roots[s + i] for i in range(M)]

    # Validate; if any root is invalid (should not happen), search exhaustively.
    for i in range(M):
        v = s + i
        rt = result_roots[i]
        if (rt * rt) % P != v:
            for xv in range(1, P):
                if (xv * xv) % P == v:
                    result_roots[i] = xv
                    break
            else:
                raise RuntimeError(f"No root found for residue {v}")

    return {'start': int(s), 'roots': [int(r) for r in result_roots]}
