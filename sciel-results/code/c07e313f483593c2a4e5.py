"""Rewritten solver: find the minimum number of distinct divisors of 5040 summing to 4979."""
import time

_N = 5040
_TARGET = 4979
_DEADLINE = 50.0  # safety margin below the 60s contract


def _divisors_of(n):
    ds = []
    for d in range(1, n + 1):
        if n % d == 0:
            ds.append(d)
    return ds


def construct():
    t0 = time.time()
    divs = _divisors_of(_N)  # 60 divisors, sorted ascending
    m = len(divs)
    target = _TARGET

    # 0/1 knapsack DP: dp[s] = min number of divisors to reach sum s
    # Also track which divisors were used for reconstruction.
    # dp[s] = (count, frozenset_of_indices) is too heavy; instead:
    # dp[s] = min count, and parent[s] = (prev_sum, divisor_index) for backtracking.
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, div_idx)
    
    # Process each divisor; for 0/1 knapsack, iterate sums in descending order
    for idx, d in enumerate(divs):
        if d > target:
            continue
        # Update in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, idx)
    
    if dp[target] == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in range(_N, 0, -1):
            if _N % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Backtrack to find which divisors were selected
    chosen = []
    cur = target
    while parent[cur] is not None:
        prev_s, div_idx = parent[cur]
        chosen.append(divs[div_idx])
        cur = prev_s
    
    # Verify
    assert sum(chosen) == target, f"Sum {sum(chosen)} != {target}"
    assert len(set(chosen)) == len(chosen), "Duplicate divisors found"
    assert all(_N % x == 0 for x in chosen), "Non-divisor found"
    assert len(chosen) <= 256, "Too many divisors"
    
    return chosen
