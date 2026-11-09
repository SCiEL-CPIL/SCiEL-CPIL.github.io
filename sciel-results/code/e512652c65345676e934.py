"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys
import random


def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']

    # Build adjacency masks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1
    size = 1 << n

    # Precompute is_clique and is_indep for all masks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        if is_clique[rest]:
            is_clique[mask] = (adj[v] & rest) == rest
        if is_indep[rest]:
            is_indep[mask] = (adj[v] & rest) == 0

    valid = [False] * size
    for m in range(1, size):
        valid[m] = is_clique[m] or is_indep[m]

    # Precompute valid submasks for each mask
    valid_subs = [[] for _ in range(size)]
    for mask in range(1, size):
        sub = mask
        while sub:
            if valid[sub]:
                valid_subs[mask].append(sub)
            sub = (sub - 1) & mask

    # DP: dp[mask] = min number of valid parts to cover mask
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    parent = [0] * size

    for mask in range(1, size):
        best = INF
        best_sub = 0
        for sub in valid_subs[mask]:
            rest = mask ^ sub
            val = dp[rest] + 1
            if val < best:
                best = val
                best_sub = sub
                # Early exit: can't do better than 1
                if best == 1:
                    break
        dp[mask] = best
        parent[mask] = best_sub

    # Check deadline
    if time.time() - t0 > DEADLINE:
        # Fallback: singletons
        return [[i] for i in range(n)]

    # Reconstruct
    result_masks = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        result_masks.append(sub)
        mask ^= sub

    result = []
    for m in result_masks:
        verts = []
        r = m
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            verts.append(v)
            r ^= lb
        result.append(verts)

    if not result:
        return [[i] for i in range(n)]

    return result
# EVOLVE-BLOCK-END
