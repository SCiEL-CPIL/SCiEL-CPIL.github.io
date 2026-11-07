import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    n = 12
    period = 27720  # lcm(1..12)
    max_L = 144

    # Precompute: for each i, valid_offsets[i][rem] = sorted list of d in 1..max_L where d % i == rem
    valid_offsets = {}
    for i in range(1, n + 1):
        valid_offsets[i] = [[] for _ in range(i)]
        for d in range(1, max_L + 1):
            rem = d % i
            valid_offsets[i][rem].append(d)

    def has_perfect_matching(m, L):
        """Check if (m, m+L] has a perfect matching for distinct multiples."""
        if L < n:
            return False

        # For each i, compute the valid d's up to L
        adj = []  # adj[i-1] = list of valid d's for left node i
        counts = []
        for i in range(1, n + 1):
            rem = (-m) % i
            opts = []
            for d in valid_offsets[i][rem]:
                if d > L:
                    break
                opts.append(d)
            adj.append(opts)
            counts.append(len(opts))

        # Check if any node has 0 options
        for c in counts:
            if c == 0:
                return False

        # Hall's condition quick check: union of all neighbors must have size >= n
        all_neighbors = set()
        for opts in adj:
            all_neighbors.update(opts)
        if len(all_neighbors) < n:
            return False

        # Order left nodes by count (ascending) - most constrained first
        order = sorted(range(n), key=lambda idx: counts[idx])

        # matchR[d] = index in original left set (0-based) that d is matched to
        matchR = [-1] * (L + 1)

        def dfs(idx, seen):
            # idx is the original left node index (0-based)
            for d in adj[idx]:
                if seen[d]:
                    continue
                seen[d] = True
                if matchR[d] == -1 or dfs(matchR[d], seen):
                    matchR[d] = idx
                    return True
            return False

        for orig_idx in order:
            seen = [False] * (L + 1)
            if not dfs(orig_idx, seen):
                return False
        return True

    best_m = 0
    best_L = 0

    # Scan all m values in one period
    for m in range(period):
        if time.time() - start_time > deadline:
            break

        # Binary search for minimum L
        lo, hi = 1, max_L
        min_L_for_m = max_L

        while lo <= hi:
            mid = (lo + hi) // 2
            if has_perfect_matching(m, mid):
                min_L_for_m = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if min_L_for_m > best_L:
            best_L = min_L_for_m
            best_m = m

    if best_m < 0 or best_m > 1000000:
        best_m = 0

    return int(best_m)
