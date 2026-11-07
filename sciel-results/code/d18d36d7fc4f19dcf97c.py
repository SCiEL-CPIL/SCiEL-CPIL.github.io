"""Rewritten program: exact cochromatic partition via bitmask DP with memoization."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    full_mask = (1 << n) - 1

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute is_clique and is_indep for all masks using DP
    size = 1 << n
    is_clique = bytearray(size)
    is_indep = bytearray(size)
    is_clique[0] = 1
    is_indep[0] = 1

    for mask in range(1, size):
        low = mask & (-mask)
        v = low.bit_length() - 1
        rest = mask ^ low
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = 1
        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = 1

    # is_valid: mask is a valid part (clique or independent set)
    is_valid = bytearray(size)
    for m in range(size):
        is_valid[m] = is_clique[m] | is_indep[m]

    # DP: dp[mask] = min number of parts to partition mask
    # Use -1 as "uncomputed"
    dp = [-1] * size
    dp[0] = 0
    best_sub = [0] * size  # store which submask was chosen

    deadline = time.time() + 85.0  # safety margin within 90s

    def solve(mask):
        if time.time() > deadline:
            return 9999
        if dp[mask] != -1:
            return dp[mask]

        best = 9999
        best_s = 0

        # Pick lowest set bit to fix symmetry: only consider submasks containing it
        low = mask & (-mask)

        sub = mask
        while sub:
            if (sub & low) and is_valid[sub]:
                rest = mask ^ sub
                val = 1 + solve(rest)
                if val < best:
                    best = val
                    best_s = sub
            sub = (sub - 1) & mask

        dp[mask] = best
        best_sub[mask] = best_s
        return best

    result_count = solve(full_mask)

    # Fallback: greedy if DP didn't finish
    if result_count >= 9999:
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                pm = 0
                for x in part:
                    pm |= (1 << x)
                trial = pm | (1 << v)
                if is_valid[trial]:
                    part.append(int(v))
                    placed = True
                    break
            if not placed:
                parts.append([int(v)])
        return parts

    # Reconstruct partition
    partition_masks = []
    mask = full_mask
    while mask:
        s = best_sub[mask]
        if s == 0:
            break
        partition_masks.append(s)
        mask ^= s

    # Convert masks to lists of int
    result = []
    for pm in partition_masks:
        verts = []
        m = pm
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            verts.append(int(v))
            m ^= low
        if verts:
            result.append(verts)

    # Validate
    all_v = []
    for part in result:
        all_v.extend(part)
    if sorted(all_v) != list(range(n)) or not result:
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                pm = 0
                for x in part:
                    pm |= (1 << x)
                trial = pm | (1 << v)
                if is_valid[trial]:
                    part.append(int(v))
                    placed = True
                    break
            if not placed:
                parts.append([int(v)])
        return parts

    return result
# EVOLVE-BLOCK-END
