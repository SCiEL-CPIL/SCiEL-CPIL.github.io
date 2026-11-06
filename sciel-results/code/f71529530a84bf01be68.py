import time
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Enumerate all non-empty subsets, group by size
    subsets_by_size = [[] for _ in range(11)]
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        size = len(terms)
        subsets_by_size[size].append((s, terms))
    
    # For each residue, find the smallest subset (by size, then by sum of denominators as tiebreaker)
    best_for_residue = {}
    
    for L in range(1, 11):
        for residue, terms in subsets_by_size[L]:
            if residue not in best_for_residue:
                best_for_residue[residue] = terms
    
    # Verify coverage
    if len(best_for_residue) < P:
        # This should not happen, but fallback
        # Use any subset for uncovered residues
        all_subsets = []
        for L in range(1, 11):
            all_subsets.extend(subsets_by_size[L])
        for residue, terms in all_subsets:
            if residue not in best_for_residue:
                best_for_residue[residue] = terms
    
    result = [best_for_residue[r] for r in range(P)]
    
    # Verify correctness
    for r in range(P):
        row = result[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has invalid denominator"
        s = sum(inv[x] for x in row) % P
        assert s == r, f"Row {r}: expected residue {r}, got {s}"
    
    return result
