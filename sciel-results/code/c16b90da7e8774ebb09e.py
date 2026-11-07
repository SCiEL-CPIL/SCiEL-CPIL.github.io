"""Exact cochromatic partition via optimized subset DP with memoized validity."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 84.0  # safety margin under 90s wall-clock
    n = P['n']

    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1
    size = 1 << n

    # valid[mask]: True iff mask induces a clique OR an independent set.
    # Memoized incremental computation: for mask with least-significant bit v,
    # let rest = mask ^ (1<<v). Then:
    #   mask is a clique      iff rest is a clique     AND rest ⊆ adj[v]
    #   mask is an indep set  iff rest is an indep set AND rest ⊆ non_adj[v]
    # This makes the whole precomputation O(2^n) instead of O(n * 2^n).
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask ^ (1 << v) ^ adj[v]

    valid = bytearray(size)
    valid[0] = 1
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        # clique: rest must be clique AND all of rest adjacent to v
        is_clique = bool(valid[rest]) and (rest & ~adj[v]) == 0
        # indep:  rest must be indep  AND all of rest non-adjacent to v
        is_indep  = bool(valid[rest]) and (rest & adj[v]) == 0
        if is_clique or is_indep:
            valid[mask] = 1

    # DP over subsets: dp[mask] = min parts to partition mask into valid parts.
    INF = 255
    dp = bytearray(size)
    dp[0] = 0
    for mask in range(1, size):
        sub = mask
        best = 255
        while sub:
            if valid[sub]:
                prev = dp[mask ^ sub]
                if prev < 255:
                    cand = prev + 1
                    if cand < best:
                        best = cand
                        if best == 1:
                            break
            sub = (sub - 1) & mask
        dp[mask] = best
        if time.time() > deadline:
            break

    # Reconstruct the partition achieving dp[full_mask].
    if dp[full_mask] < 255:
        parts = []
        mask = full_mask
        while mask:
            sub = mask
            found = False
            while sub:
                if valid[sub]:
                    prev = dp[mask ^ sub]
                    if prev + 1 == dp[mask]:
                        part = [v for v in range(n) if (sub >> v) & 1]
                        parts.append(part)
                        mask ^= sub
                        found = True
                        break
                sub = (sub - 1) & mask
            if not found:
                # Should not happen; keep validity as a last resort.
                parts.append([v for v in range(n) if (mask >> v) & 1])
                break
        return parts
    else:
        # Fallback greedy (valid partition) in case DP was cut off.
        es = set()
        for a, b in P['edges']:
            es.add((a, b))
            es.add((b, a))
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
# EVOLVE-BLOCK-END
