"""Cochromatic partition solver: memoized DP with branch-and-bound refinement."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    full_mask = (1 << n) - 1

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute is_clique and is_ind_set for all masks using incremental check
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest]:
            # v must be adjacent to all vertices in rest
            if (adj[v] & rest) == rest:
                is_clique[mask] = True

        if is_ind[rest]:
            # v must be non-adjacent to all vertices in rest
            if (adj[v] & rest) == 0:
                is_ind[mask] = True

    # Build list of all valid non-empty masks (clique or independent set)
    valid_masks = [m for m in range(1, 1 << n) if is_clique[m] or is_ind[m]]

    # Sort valid masks by size descending for better pruning
    valid_masks.sort(key=lambda m: m.bit_count(), reverse=True)

    # Memoized DP: minimum number of parts to partition mask
    # dp[mask] = min parts needed, or -1 if unknown
    dp = [-1] * (1 << n)
    dp[0] = 0

    # choice[mask] = the part chosen for optimal partition of mask
    choice = [0] * (1 << n)

    # Compute dp bottom-up by increasing popcount
    # For each mask, try all valid subsets S of mask
    # dp[mask] = 1 + min(dp[mask ^ S]) over valid S subset of mask
    # To make it efficient, we only need S to contain the lowest set bit of mask

    # Process masks in order of increasing size
    masks_by_size = list(range(1, 1 << n))
    masks_by_size.sort(key=lambda m: m.bit_count())

    for mask in masks_by_size:
        if time.time() > deadline:
            break

        # If mask itself is valid, dp = 1
        if is_clique[mask] or is_ind[mask]:
            dp[mask] = 1
            choice[mask] = mask
            continue

        # Pick lowest set bit to anchor our search
        lsb = mask & -mask

        best = n + 1
        best_part = 0

        # Try all valid masks S that are subsets of mask and contain lsb
        for S in valid_masks:
            if (S & mask) != S:
                continue
            if (S & lsb) == 0:
                continue
            # S must be a proper subset (since mask itself isn't valid)
            rem = mask ^ S
            if dp[rem] >= 0 and dp[rem] + 1 < best:
                best = dp[rem] + 1
                best_part = S
                if best == 2:  # can't do better than 2 for non-valid mask
                    break

        if best <= n:
            dp[mask] = best
            choice[mask] = best_part
        else:
            dp[mask] = n  # fallback: singleton partition

    if time.time() > deadline:
        # Fallback: greedy partition
        return _greedy_partition(n, adj)

    # Reconstruct partition from choice table
    parts = []
    mask = full_mask
    while mask != 0:
        S = choice[mask]
        part_verts = [v for v in range(n) if (S >> v) & 1]
        parts.append(part_verts)
        mask ^= S

    # Ensure types: list of lists of Python ints
    return [[int(v) for v in part] for part in parts]


def _greedy_partition(n, adj):
    """Greedy fallback: assign each vertex to existing part if possible, else new part."""
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            # Check if adding v keeps part as clique or independent set
            is_c = True
            is_i = True
            for u in part:
                if (adj[u] >> v) & 1:
                    is_i = False
                else:
                    is_c = False
            if is_c or is_i:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return [[int(v) for v in part] for part in parts]

# EVOLVE-BLOCK-END
