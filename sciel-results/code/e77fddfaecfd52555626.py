"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_START = time.time()
_DEADLINE = 80.0  # safety margin under the 90s kill limit


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Precompute subset sums for all subsets, grouped by size
    subset_sums = [[] for _ in range(N + 1)]

    for size in range(1, N + 1):
        for terms in combinations(range(1, N + 1), size):
            residue = sum(inv[x] for x in terms) % MOD
            subset_sums[size].append((residue, list(terms)))

    # Binary search on maximum allowed subset size
    # Check if all 101 residues can be covered using subsets of size <= k
    def can_cover(max_size):
        """Try to cover all residues using subsets of size <= max_size.
        Returns assignment dict if successful, None otherwise."""
        if _elapsed() > _DEADLINE:
            return None
        
        # Collect all candidate subsets up to max_size
        candidates = []
        for size in range(1, max_size + 1):
            candidates.extend(subset_sums[size])
        
        # Greedy: assign smallest subsets first to uncovered residues
        # Sort by size, then by some tiebreaker
        candidates_sorted = sorted(candidates, key=lambda x: (len(x[1]), x[0]))
        
        best = [None] * MOD
        for residue, terms in candidates_sorted:
            if best[residue] is None:
                best[residue] = terms
        
        if all(b is not None for b in best):
            return best
        return None

    # Try sizes from 1 upward, find minimum max_size that works
    best_result = None
    for max_size in range(1, N + 1):
        if _elapsed() > _DEADLINE:
            break
        result = can_cover(max_size)
        if result is not None:
            best_result = result
            break

    # If no size worked (shouldn't happen since size 10 always works), fall back
    if best_result is None:
        best_result = [None] * MOD
        for size in range(1, N + 1):
            if _elapsed() > _DEADLINE:
                break
            for residue, terms in subset_sums[size]:
                if best_result[residue] is None:
                    best_result[residue] = terms

    return [best_result[r] for r in range(MOD)]
# EVOLVE-BLOCK-END
