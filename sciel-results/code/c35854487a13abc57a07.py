"""Minimal run-start s for 5 consecutive non-zero quadratic residues mod 1009.

r(k,m,p) with k=2 (quadratic), m=5 (run length), p=1009.
Return {'start':s,'roots':[r0..r4]} where r_i^2 mod 1009 == s+i, s in [1,1004].
Minimize s; zero residues and wraparound excluded.
"""

import time

_P = 1009
_M = 5
_DEADLINE = 85.0  # seconds; safety margin under 90s kill


def construct():
    start_time = time.time()

    # Build a list mapping each non-zero quadratic residue to one root in [1, P-1].
    # We only need one root per residue for certification.
    root_map = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in root_map:
            root_map[r] = x

    # Build a bitset: bit v is 1 iff v is a non-zero quadratic residue.
    # We use Python's arbitrary-precision integers for fast bitwise ops.
    res_bits = 0
    for v in range(1, _P):
        if v in root_map:
            res_bits |= (1 << v)

    # Find all positions s where s, s+1, s+2, s+3, s+4 are all residues.
    # Bit s is set in matches iff a run of 5 consecutive residues starts at s.
    matches = res_bits & (res_bits >> 1) & (res_bits >> 2) & (res_bits >> 3) & (res_bits >> 4)

    # Find the smallest s >= 1 such that bit s is set in matches.
    # Also require s <= P - 5 = 1004 (so s+4 <= 1008, no wraparound into 0 or P).
    if matches:
        # Iterate through set bits from least significant to most.
        # We want the smallest s in [1, 1004].
        temp = matches
        s = -1
        while temp:
            # Get the index of the least significant set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            if 1 <= idx <= _P - _M:
                s = idx
                break
            # Clear the least significant set bit
            temp ^= lsb

        if s >= 0:
            roots = [root_map[s + i] for i in range(_M)]
            # Verify correctness
            for i, r in enumerate(roots):
                assert (r * r) % _P == s + i, f"Verification failed: {r}^2 mod {_P} != {s+i}"
            return {'start': s, 'roots': roots}

    # Fallback: linear scan (should not be reached for p=1009)
    is_res = [False] * _P
    for v in range(1, _P):
        if v in root_map:
            is_res[v] = True

    for s in range(1, _P - _M + 1):
        if time.time() - start_time > _DEADLINE:
            raise RuntimeError("Deadline exceeded")
        if all(is_res[s + i] for i in range(_M)):
            roots = [root_map[s + i] for i in range(_M)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
