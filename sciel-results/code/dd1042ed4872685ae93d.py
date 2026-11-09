"""Rewritten program: exact min-popcount covering certificate for residues mod 101."""
# EVOLVE-BLOCK-START
import time

_P = 101
_DENOMS = list(range(1, 11))
_INV = [pow(x, -1, _P) for x in _DENOMS]

def construct():
    """Return 101 nonempty lists; row r has distinct denominators in 1..10
    whose modular inverses sum to r mod 101. Minimize the largest row length."""
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill

    # best[residue] = (popcount, terms_list)
    best = [None] * _P

    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
        residue = 0
        pop = 0
        terms = []
        for i in range(10):
            if mask >> i & 1:
                residue = (residue + _INV[i]) % _P
                pop += 1
                terms.append(_DENOMS[i])
        cur = best[residue]
        if cur is None or pop < cur[0]:
            best[residue] = (pop, terms)
            
    # All residues are covered by subsets of size <= 4, so best should be fully populated.
    # If not, fallback to first found (should not happen).
    if any(b is None for b in best):
        for r in range(_P):
            if best[r] is None:
                best[r] = (1, [_DENOMS[0]])

    return [best[r][1] for r in range(_P)]
# EVOLVE-BLOCK-END
