import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full = (1 << n) - 1
    deadline = time.time() + 50.0  # safety margin under 60s

    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute valid[S]: True if S induces a clique or an independent set
    size = 1 << n
    valid = bytearray(size)
    valid[0] = 1
    for S in range(1, size):
        # find a vertex in S
        lsb = S & (-S)
        v = lsb.bit_length() - 1
        rest = S ^ lsb
        if not valid[rest]:
            continue
        # Check if S is a clique: v must be adjacent to every vertex in rest
        if rest & ~adj[v] == 0:
            valid[S] = 1
            continue
        # Check if S is an independent set: v must be non-adjacent to every vertex in rest
        if rest & adj[v] == 0:
            valid[S] = 1
            continue
        # Otherwise S is neither

    # DP: dp[S] = min parts to partition S; parent[S] = a valid sub-part T of S used
    INF = 1 << 30
    dp = bytearray([255]) * size  # 255 as INF marker (max parts = 14)
    parent = [-1] * size
    dp[0] = 0

    # Iterate by popcount (number of set bits)
    # For each subset S, try each valid sub-part T (T subset of S, T nonempty)
    # dp[S] = min over valid T subset of S of dp[S ^ T] + 1

    # Group subsets by popcount for ordering
    from collections import defaultdict
    by_popcount = defaultdict(list)
    for S in range(1, size):
        by_popcount[S.bit_count()].append(S)

    for k in range(1, n + 1):
        for S in by_popcount[k]:
            best = INF
            best_T = -1
            # Enumerate nonempty sub-subsets T of S
            T = S
            while T:
                if valid[T]:
                    rem = S ^ T
                    val = dp[rem] + 1
                    if val < best:
                        best = val
                        best_T = T
                T = (T - 1) & S
            dp[S] = best
            parent[S] = best_T

    # Check if we completed
    if dp[full] < 255:
        # Reconstruct partition
        parts = []
        S = full
        while S:
            T = parent[S]
            if T == -1:
                break
            part = [i for i in range(n) if (T >> i) & 1]
            parts.append(part)
            S ^= T
        # Validate
        es = {tuple(e) for e in P['edges']}
        for part in parts:
            if not part:
                break
            hits = [(u, w) in es or (w, u) in es
                    for i, u in enumerate(part) for w in part[i+1:]]
            if not (all(hits) or not any(hits)):
                break
        else:
            # All parts valid; ensure all vertices covered exactly once
            seen = [False] * n
            for part in parts:
                for v in part:
                    if seen[v]:
                        break
                    seen[v] = True
            else:
                if all(seen):
                    return parts
    # Fallback: greedy (shouldn't normally reach here within budget)
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es
                    for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts
# EVOLVE-BLOCK-END
