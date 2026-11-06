"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s wall-clock kill
    
    # Precompute modular inverses of 1..10 mod 101
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # For each residue, track the minimum subset size found
    # and the corresponding subset
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Enumerate all non-empty subsets of {1,...,10} by size
    # For each subset, if it achieves a residue with smaller size, update
    from itertools import combinations
    
    for size in range(1, 11):
        if time.time() > deadline:
            break
        # Early termination: if all residues are covered by subsets of this size or smaller
        # (i.e., min_len[r] <= size for all r, meaning all are covered)
        # Actually we need to check if all are covered at all
        all_covered = all(min_len[r] < 11 for r in range(p))
        if all_covered:
            break
        
        for comb in combinations(range(1, 11), size):
            s = sum(inv[x] for x in comb) % p
            if size < min_len[s]:
                min_len[s] = size
                best_subsets[s] = list(comb)
    
    # Safety: if any residue still missing (shouldn't happen), fill with any valid subset
    if any(best_subsets[r] is None for r in range(p)):
        # Fallback: full enumeration
        reps = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            residue = sum(inv[x] for x in terms) % p
            if residue not in reps:
                reps[residue] = terms
        for r in range(p):
            if best_subsets[r] is None:
                best_subsets[r] = reps[r]
    
    return [best_subsets[r] for r in range(p)]
# EVOLVE-BLOCK-END
