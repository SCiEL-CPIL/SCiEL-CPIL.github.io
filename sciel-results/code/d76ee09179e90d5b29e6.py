"""Cochromatic partition via optimized bitmask DP."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute: for each subset, is it a clique or independent set?
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True

        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = True

    # A part is valid if it's a clique or independent set
    # Note: Empty set is not a valid part in the partition context (nonempty lists)
    # But for DP, we only look at non-empty submasks.
    is_valid_part = [False] * (1 << n)
    for m in range(1, 1 << n):
        is_valid_part[m] = is_clique[m] or is_indep[m]

    # DP: dp[mask] = minimum number of parts to cover vertices in mask
    # parent[mask] = the submask 'sub' such that dp[mask] = dp[mask ^ sub] + 1 and sub is valid
    INF = 100
    dp = [INF] * (1 << n)
    parent = [-1] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        best = INF
        best_sub = -1
        # Enumerate all submasks of mask
        sub = mask
        while sub > 0:
            if is_valid_part[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_sub = sub
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best
        parent[mask] = best_sub

    min_parts = dp[full_mask]

    # Reconstruct the partition
    parts_masks = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            # Should not happen if dp is correct, but fallback
            sub = mask
        parts_masks.append(sub)
        mask = mask ^ sub

    # Convert bitmask parts to lists of vertices
    result = []
    for part_mask in parts_masks:
        verts = []
        for v in range(n):
            if part_mask & (1 << v):
                verts.append(v)
        result.append(verts)

    # Validate the result
    def validate(parts):
        seen = [False] * n
        for part in parts:
            if not part:
                return False
            for v in part:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if seen[v]:
                    return False
                seen[v] = True
            # Check clique or independent
            is_c = True
            is_i = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if adj[part[i]] & (1 << part[j]):
                        is_i = False
                    else:
                        is_c = False
            if not (is_c or is_i):
                return False
        return all(seen)

    if not validate(result):
        # Fallback: all singletons (always valid)
        result = [[v] for v in range(n)]

    return result
