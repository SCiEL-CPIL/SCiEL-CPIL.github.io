"""
Finds the minimal starting value s such that s, s+1, ..., s+4 are all non-zero
quadratic residues modulo 1009. Returns s and the corresponding square roots.

Strategy:
1. Precompute a lookup table for quadratic residues and their roots.
2. Use a bitmask representation to efficiently check for consecutive runs of residues.
3. Scan for the first occurrence of a run of length 5.
"""

import time

P = 1009
M = 5

def construct():
    # 1. Precompute roots for non-zero quadratic residues
    # qr_root[v] = r such that r*r % P == v, or 0 if v is not a non-zero QR
    qr_root = [0] * P
    for x in range(1, P):
        r = (x * x) % P
        if r != 0 and qr_root[r] == 0:
            qr_root[r] = x

    # 2. Build a bitmask for non-zero quadratic residues
    # Bit v is set to 1 if v is a non-zero quadratic residue
    mask = 0
    for v in range(1, P):
        if qr_root[v] != 0:
            mask |= (1 << v)

    # 3. Find the minimal s in [1, P-M] such that s..s+4 are all QRs
    target_pattern = (1 << M) - 1

    for s in range(1, P - M + 1):
        if ((mask >> s) & target_pattern) == target_pattern:
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}

    # Fallback (should not trigger)
    for s in range(1, P - M + 1):
        if all(qr_root[s + i] != 0 for i in range(M)):
            roots = [qr_root[s + i] for i in range(M)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive nonzero QRs found')
