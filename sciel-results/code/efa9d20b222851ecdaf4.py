"""Optimized cochromatic partition using DP over valid subsets with deadline guard."""

import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill

    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Precompute which subsets are cliques and which are independent sets
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)

    # Empty set and singletons are both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    for i in range(n):
        mask = 1 << i
        is_clique[mask] = True
        is_indep[mask] = True

    # Fill DP for clique/independent property
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True

        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True

    # Check deadline
    if time.time() > deadline:
        # Fallback: each vertex as its own part (valid: singletons are both clique and indep)
        return [[i] for i in range(n)]

    # DP for minimum number of parts
    # dp[mask] = min parts to partition mask
    # choice[mask] = the subset T chosen as the first part
    dp = [float('inf')] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0

    for mask in range(1, 1 << n):
        sub = mask
        best_val = dp[mask]
        best_sub = choice[mask]
        while sub:
            if is_clique[sub] or is_indep[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        dp[mask] = best_val
        choice[mask] = best_sub

        # Periodic deadline check (every 256 masks)
        if (mask & 255) == 0 and time.time() > deadline:
            break

    # Check if we completed the DP
    full_mask = (1 << n) - 1
    if dp[full_mask] == float('inf'):
        # DP didn't complete; fallback to singleton partition
        return [[i] for i in range(n)]

    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        part_mask = choice[current_mask]
        if part_mask == 0:
            # Should not happen, but guard against it
            return [[i] for i in range(n)]
        part_vertices = []
        for i in range(n):
            if (part_mask >> i) & 1:
                part_vertices.append(i)
        parts.append(part_vertices)
        current_mask ^= part_mask

    # Convert to list of lists of ints
    result = [list(p) for p in parts]
    for part in result:
        for i in range(len(part)):
            part[i] = int(part[i])

    return result
