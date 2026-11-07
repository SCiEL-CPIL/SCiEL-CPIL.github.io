"""Constructive algorithm: greedily add smallest valid integer, verifying
cardinality-separation via exact subset-sum DP per cardinality."""
import time

P = {'n': 256, 'maximum_size': 60}

def _satisfies(S):
    """Check whether S satisfies the cardinality-separation constraint:
    for every k, min (k+1)-subset sum > max k-subset sum.
    Uses subset-sum DP per cardinality. Empty subset sum = 0.
    """
    n = len(S)
    if n == 0:
        return True
    max_sum = sum(S)
    # dp[k][s] = True if some k-subset sums to s
    # We only need min and max reachable sum per cardinality.
    # But to be safe, compute full reachability per cardinality (bitsets).
    # dp[k] is a bitset (int) where bit s is set iff s is reachable by k-subset.
    # Start: dp[0] has bit 0 set.
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0
    for x in S:
        # process in reverse so each x used at most once
        for k in range(n, 0, -1):
            if dp[k - 1]:
                dp[k] |= (dp[k - 1] << x)
    # For each k, find min and max reachable sum.
    # min: lowest set bit; max: highest set bit.
    prev_max = -1
    for k in range(0, n):
        # max of k-subset
        if dp[k] == 0:
            # no k-subset? (shouldn't happen for k<=n)
            return False
        # highest set bit
        max_k = dp[k].bit_length() - 1
        # min of (k+1)-subset
        if dp[k + 1] == 0:
            return False
        # lowest set bit
        min_k1 = (dp[k + 1] & -dp[k + 1]).bit_length() - 1
        if not (min_k1 > max_k):
            return False
        prev_max = max_k
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin under 60s
    S = []
    # Greedily try to add integers 1..256 in increasing order.
    # We want to maximize |S| up to 60.
    # Strategy: iterate candidate c from 1..256; if S + [c] satisfies and |S|<60, add it.
    # But greedy by smallest may not maximize size. Better: try to reach size 60.
    # Approach: build a set by considering candidates in increasing order,
    # but we may skip some to allow more later. Simple greedy first; if it stalls,
    # try a backtracking / local search.
    
    # First pass: pure greedy smallest-first.
    for c in range(1, 257):
        if len(S) >= 60:
            break
        if time.time() > deadline - 5.0:
            break
        trial = S + [c]
        if _satisfies(trial):
            S = trial
    if len(S) >= 60:
        return list(S)
    
    # If greedy didn't reach 60, try a smarter search:
    # Local search: try to swap out a larger element for a smaller unused one,
    # or add a new element by removing a blocking one.
    # Represent S as sorted list.
    S = sorted(S)
    used = set(S)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to add each unused candidate in increasing order.
        for c in range(1, 257):
            if c in used:
                continue
            trial = S + [c]
            if _satisfies(trial):
                S = sorted(trial)
                used.add(c)
                improved = True
                if len(S) >= 60:
                    return list(S)
                break
        # If we couldn't add, try removing the largest element and re-adding smaller.
        if len(S) > 0:
            big = S[-1]
            S2 = S[:-1]
            used2 = set(S2)
            for c in range(1, big):
                if c in used2:
                    continue
                trial = S2 + [c]
                if _satisfies(trial):
                    S = sorted(trial)
                    used = set(S)
                    improved = True
                    break
    
    return list(S)
