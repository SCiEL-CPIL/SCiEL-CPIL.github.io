"""Rewritten program: exact min-popcount covering certificate for residues mod 101."""
# EVOLVE-BLOCK-START
import time

_P = 101
_DENOMS = list(range(1, 11))
_INV = [pow(x, -1, _P) for x in _DENOMS]
_NDEN = 10
_NMASKS = 1 << _NDEN

def construct():
    """Return 101 nonempty lists; row r has distinct denominators in 1..10
    whose modular inverses sum to r mod 101. Minimize the largest row length."""
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill

    # Precompute residue and popcount for each mask
    # best[residue] = (popcount, terms_list)
    best = [None] * _P

    for mask in range(1, _NMASKS):
        residue = 0
        pop = 0
        terms = []
        m = mask
        i = 0
        while m:
            if m & 1:
                residue = (residue + _INV[i]) % _P
                pop += 1
                terms.append(_DENOMS[i])
            i += 1
            m >>= 1
        cur = best[residue]
        if cur is None or pop < cur[0]:
            best[residue] = (pop, terms)

    # Verify all residues are covered
    if any(b is None for b in best):
        for r in range(_P):
            if best[r] is None:
                best[r] = (1, [_DENOMS[0]])

    # Final validation: ensure each row is valid
    result = [best[r][1] for r in range(_P)]
    for r, terms in enumerate(result):
        s = sum(_INV[d - 1] for d in terms) % _P
        assert s == r, f"Row {r}: got sum {s}, expected {r}"
        assert len(terms) == len(set(terms)), f"Row {r}: duplicate denominators"
        assert all(1 <= d <= 10 for d in terms), f"Row {r}: out of range"
        assert len(terms) > 0, f"Row {r}: empty"

    return result
# EVOLVE-BLOCK-END
