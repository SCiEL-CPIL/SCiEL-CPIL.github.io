import time
import numpy as np

def construct():
    P = 1009
    M = 5
    start_time = time.time()
    deadline = start_time + 55.0

    # Build roots array: roots[r] = smallest x in 1..P-1 such that x*x % P == r
    # Use a plain list for speed (P is small)
    roots = [0] * P
    # Also build a bitset of which residues are QRs
    qr_bits = 0

    for x in range(1, P):
        r = (x * x) % P
        if roots[r] == 0:
            roots[r] = x
        qr_bits |= (1 << r)

    # Mask for M consecutive bits
    mask = (1 << M) - 1

    # Search for the minimum s such that s..s+M-1 are all QRs
    # s ranges from 1 to P-M (i.e., 1 to 1004)
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            break
        # Check if bits s, s+1, ..., s+M-1 are all set
        if ((qr_bits >> s) & mask) == mask:
            current_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': current_roots}

    # Fallback (should not happen for p=1009)
    return {'start': P - M, 'roots': [1, 1, 1, 1, 1]}
