"""Minimal run-start s for 5 consecutive non-zero quadratic residues mod 1009.

r(k,m,p) with k=2 (quadratic), m=5 (run length), p=1009.
Return {'start':s,'roots':[r0..r4]} where r_i^2 mod 1009 == s+i, s in [1,1004].
Minimize s; zero residues and wraparound excluded.
"""
import time

_P = 1009
_START = time.time()
_DEADLINE = 90.0
_MARGIN = 5.0


def _elapsed():
    return time.time() - _START


def construct():
    # Map residue -> one root r in [1, P-1] with r^2 % P == residue (skip residue 0).
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and root[r] == 0:
            root[r] = x

    # Build a bitset: bit v is 1 iff v (1..1008) is a non-zero quadratic residue.
    # Bit position v corresponds to residue v.
    bitmask = 0
    for v in range(1, _P):
        if root[v] != 0:
            bitmask |= (1 << v)

    # To find 5 consecutive residues starting at s, we need bits s, s+1, s+2, s+3, s+4 all set.
    # This is equivalent to: (bitmask >> s) & 0b11111 == 0b11111
    # Scan s from 1 upward; require s+4 <= 1008 (s <= 1004) to avoid wraparound.
    # First hit is the minimal s.
    max_s = _P - 5  # 1004
    for s in range(1, max_s + 1):
        if (bitmask >> s) & 0b11111 == 0b11111:
            roots = [root[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}

        # Deadline guard: never run past the wall-clock limit.
        if _elapsed() > _DEADLINE - _MARGIN:
            break

    raise RuntimeError('No 5-run of non-zero quadratic residues found modulo 1009')
