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

    # Precompute: for each size, map residue -> list of term-lists
    subset_by_size = [dict() for _ in range(N + 1)]

    for size in range(1, N + 1):
        for terms in combinations(range(1, N + 1), size):
            residue = sum(inv[x] for x in terms) % MOD
            if residue not in subset_by_size[size]:
                subset_by_size[size][residue] = []
            subset_by_size[size][residue].append(list(terms))

    # Greedy by size, with most-constrained-first within each size
    best = [None] * MOD
    max_size_used = 0

    for size in range(1, N + 1):
        if _elapsed() > _DEADLINE:
            break

        # Collect uncovered residues with options at this size
        candidates = []
        for r in range(MOD):
            if best[r] is None and r in subset_by_size[size]:
                candidates.append((len(subset_by_size[size][r]), r))

        # Most constrained first (fewest options)
        candidates.sort()

        for _, r in candidates:
            if best[r] is None:
                best[r] = subset_by_size[size][r][0]
                max_size_used = size

        if all(b is not None for b in best):
            break

    # Fallback fill if not all covered
    if any(b is None for b in best):
        for size in range(1, N + 1):
            if _elapsed() > _DEADLINE:
                break
            for r in range(MOD):
                if best[r] is None and r in subset_by_size[size]:
                    best[r] = subset_by_size[size][r][0]
                    if size > max_size_used:
                        max_size_used = size

    # Local improvement: try to reduce max row length
    # Compute current max
    cur_max = max(len(b) for b in best if b is not None)

    # For each row at cur_max, try to find a smaller subset
    improved = True
    while improved and _elapsed() < _DEADLINE:
        improved = False
        for r in range(MOD):
            if best[r] is not None and len(best[r]) == cur_max:
                for ss in range(1, cur_max):
                    if r in subset_by_size[ss]:
                        best[r] = subset_by_size[ss][r][0]
                        cur_max = max(len(b) for b in best if b is not None)
                        improved = True
                        break

    return [best[r] for r in range(MOD)]
# EVOLVE-BLOCK-END
