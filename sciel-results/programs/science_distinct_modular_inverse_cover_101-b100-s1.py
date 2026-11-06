"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_INV = {x: pow(x, -1, 101) for x in range(1, 11)}

# Precompute all subsets: mask -> (residue, size, terms_tuple)
_SUBSETS = []  # list of (size, residue, terms_tuple)
for mask in range(1, 1 << 10):
    terms = tuple(x for x in range(1, 11) if (mask >> (x - 1)) & 1)
    r = 0
    for x in terms:
        r += _INV[x]
    r %= 101
    _SUBSETS.append((len(terms), r, terms))

def construct():
    deadline = time.time() + 60
    
    # Group subsets by size
    by_size = [[] for _ in range(11)]  # index 1..10
    for size, r, terms in _SUBSETS:
        by_size[size].append((r, terms))
    
    # Try increasing max size k
    for k in range(1, 11):
        if time.time() > deadline - 2:
            break
        # Collect all residues achievable with size <= k
        residues = {}
        for s in range(1, k + 1):
            for r, terms in by_size[s]:
                if r not in residues:
                    residues[r] = terms
        if len(residues) == 101:
            result = []
            for r in range(101):
                result.append(list(residues[r]))
            return result
    
    # Fallback: use all subsets
    representations = {}
    for size, r, terms in _SUBSETS:
        if r not in representations:
            representations[r] = list(terms)
    return [representations[r] for r in range(101)]
# EVOLVE-BLOCK-END
