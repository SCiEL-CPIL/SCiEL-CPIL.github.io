"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 85.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Enumerate all non-empty subsets of {1..10}
    # For each subset, compute (size, residue, mask)
    # There are 2^10 - 1 = 1023 subsets, very fast
    subsets = []  # list of (size, residue, mask)
    for mask in range(1, 1 << N):
        size = 0
        s = 0
        m = mask
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
        residue = s % MOD
        subsets.append((size, residue, mask))

    # Sort by size so we process smallest subsets first
    subsets.sort(key=lambda t: t[0])

    # For each residue, record the minimum-size subset that achieves it
    best_for_residue = {}  # residue -> (size, mask)
    for size, residue, mask in subsets:
        if residue not in best_for_residue:
            best_for_residue[residue] = (size, mask)

    # All 101 residues should be covered
    # Find the theoretical minimum k (max row length)
    max_size = 0
    for r in range(MOD):
        if r in best_for_residue:
            sz = best_for_residue[r][0]
            if sz > max_size:
                max_size = sz
        else:
            # Should not happen, but track it
            max_size = N

    # Build rows using the minimum-size subset for each residue
    rows = [None] * MOD
    for r in range(MOD):
        if r in best_for_residue:
            size, mask = best_for_residue[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows[r] = terms
        else:
            # Fallback (should never trigger)
            rows[r] = list(range(1, N + 1))

    return rows

# EVOLVE-BLOCK-END
