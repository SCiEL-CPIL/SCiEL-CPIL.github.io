"""Optimized: find minimum-count subset of distinct divisors of 5040 summing to 4979.
Uses meet-in-the-middle for exact minimum-count solution, with DP fallback."""
import time
from itertools import combinations

_DEADLINE = 80.0  # safety margin under 90s kill

def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort(reverse=True)
    return divs

def _verify(chosen, n, target):
    """Verify that chosen is a valid set of distinct divisors of n summing to target."""
    if len(chosen) > 256:
        return False
    seen = set()
    total = 0
    for d in chosen:
        if d in seen:
            return False
        if d <= 0 or n % d != 0:
            return False
        seen.add(d)
        total += d
    return total == target

def _meet_in_middle(divs, target, deadline):
    """
    Meet-in-the-middle: split divisors into two halves, enumerate all subsets of each,
    then for each subset of left half, look for complement in right half.
    Returns minimum-count solution or None.
    """
    n_divs = len(divs)
    if n_divs == 0:
        return None
    
    mid = n_divs // 2
    left = divs[:mid]
    right = divs[mid:]
    
    # Enumerate all subsets of left half
    # For each subset, store (sum, count, bitmask)
    left_subsets = {}  # sum -> list of (count, indices)
    for mask in range(1 << len(left)):
        s = 0
        cnt = 0
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
                cnt += 1
        if s > target:
            continue
        if s not in left_subsets:
            left_subsets[s] = []
        left_subsets[s].append((cnt, mask))
    
    # Enumerate all subsets of right half
    right_subsets = {}  # sum -> list of (count, indices)
    for mask in range(1 << len(right)):
        s = 0
        cnt = 0
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
                cnt += 1
        if s > target:
            continue
        if s not in right_subsets:
            right_subsets[s] = []
        right_subsets[s].append((cnt, mask))
    
    # For each possible total count k from 1 upward, check if we can form target
    # with exactly k elements total
    min_count = float('inf')
    best_solution = None
    
    # Sort left subsets by count for efficiency
    # For each left subset with count cl and sum sl, we need right subset with count cr = k - cl and sum sr = target - sl
    
    # Build a lookup: for right, for each (sum, count) -> indices
    # Actually, let's just iterate and track min total count
    
    for sl, left_entries in left_subsets.items():
        sr = target - sl
        if sr < 0:
            continue
        if sr not in right_subsets:
            continue
        for (cl, lmask) in left_entries:
            for (cr, rmask) in right_subsets[sr]:
                total_cnt = cl + cr
                if total_cnt == 0:
                    continue
                if total_cnt < min_count:
                    min_count = total_cnt
                    # Reconstruct
                    chosen = []
                    for i in range(len(left)):
                        if lmask & (1 << i):
                            chosen.append(left[i])
                    for i in range(len(right)):
                        if rmask & (1 << i):
                            chosen.append(right[i])
                    best_solution = chosen
    
    if best_solution is not None:
        return best_solution
    return None

def _dp_min_count(divs, target, deadline):
    """DP: dp[s] = min count to reach sum s using 0/1 knapsack. Returns solution or None."""
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    # parent tracking: parent[s] = (divisor_used, prev_sum)
    parent = [(-1, -1)] * (target + 1)
    
    for d in divs:
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (d, s - d)
    
    if dp[target] == INF:
        return None
    
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d, prev_s = parent[s]
        if d <= 0:
            break
        chosen.append(d)
        s = prev_s
    
    if _verify(chosen, 5040, target):
        return chosen
    return None

def _bfs_by_count(divs, target, n, deadline):
    """BFS by number of divisors: try all subsets of size k for k=1,2,..."""
    from itertools import combinations
    divs_sorted = sorted(divs, reverse=True)
    for k in range(1, 257):
        if time.time() > deadline:
            break
        # Prune: if sum of k largest divisors < target, skip
        if sum(divs_sorted[:k]) < target:
            continue
        for combo in combinations(divs_sorted, k):
            if time.time() > deadline:
                break
            if sum(combo) == target:
                return list(combo)
    return []

def construct():
    start = time.time()
    target = 4979
    n = 5040
    divs = _all_divisors(n)
    # only keep divisors <= target
    divs = [d for d in divs if d <= target]
    
    # Quick check: is target itself a divisor?
    if n % target == 0:
        return [target]
    
    # Try meet-in-the-middle first (exact, fast for ~60 divisors)
    if len(divs) <= 40:
        result = _meet_in_middle(divs, target, _DEADLINE)
        if result and _verify(result, n, target):
            return result
    elif len(divs) <= 50:
        # Split into two groups of ~25 each for MITM
        # We need to be careful with memory; 2^25 is too much
        # Use DP instead for this size
        pass
    
    # Use DP for larger sets
    result = _dp_min_count(divs, target, _DEADLINE)
    if result and _verify(result, n, target):
        return result
    
    # Fallback: BFS by count
    result = _bfs_by_count(divs, target, n, _DEADLINE)
    if result and _verify(result, n, target):
        return result
    
    # Last resort: greedy
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen
