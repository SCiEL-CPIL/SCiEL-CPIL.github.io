import time
from bisect import bisect_right

P = 27720  # LCM(1..12)
MAX_L = 144


def precompute_offsets():
    """Precompute for each residue r and each i, the sorted list of offsets j in 0..MAX_L-1 where i divides (r+j) mod P."""
    offsets = [[None] * 12 for _ in range(P)]
    for r in range(P):
        for i in range(12):
            lst = []
            for j in range(MAX_L):
                if (r + j) % (i + 1) == 0:
                    lst.append(j)
            offsets[r][i] = lst
    return offsets


OFFSETS = precompute_offsets()


def has_perfect_matching(m, L):
    """Check if (m, m+L] has a perfect matching for 1..12."""
    base_res = (m + 1) % P

    # Build adjacency: for each i, list of positions j < L where i+1 divides base+j
    adj = []
    for i in range(12):
        full_list = OFFSETS[base_res][i]
        idx = bisect_right(full_list, L - 1)
        adj.append(full_list[:idx])

    # Bipartite matching using DFS (Kuhn's algorithm)
    # Process left nodes in order of increasing degree (most constrained first)
    order = sorted(range(12), key=lambda i: len(adj[i]))

    match_to = [-1] * L
    match_from = [-1] * 12

    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                match_from[i] = j
                return True
        return False

    for i in order:
        visited = [False] * L
        if not dfs(i, visited):
            return False

    return True


def compute_f(m):
    """Compute f(12, m) = least L such that (m, m+L] has perfect matching."""
    lo, hi = 12, 144
    if not has_perfect_matching(m, hi):
        hi = 200
        if not has_perfect_matching(m, hi):
            return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    best_m = 0
    best_L = 0

    # First pass: scan all residues 0..P-1
    for m in range(P):
        if time.time() > deadline:
            break
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
            if best_L >= 144:
                return int(best_m)

    # If we have time, try to extend the search to cover up to 1000000
    # Since f(12, m) depends only on m mod P, any m with m % P == best_m % P
    # gives the same f value. But we need m in [0, 1000000].
    # The best_m found is already in [0, P-1] which is in [0, 1000000].
    # So we're done.

    return int(best_m)
