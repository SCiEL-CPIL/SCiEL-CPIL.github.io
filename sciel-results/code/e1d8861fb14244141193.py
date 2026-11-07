"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i using Kuhn's algorithm."""
    n = 12
    # Build adjacency: for each i (1..12), list of offsets j in [0, L-1] such that i | (m+1+j)
    adj = [[] for _ in range(n)]
    
    for i in range(1, n + 1):
        # We need k*i in (m, m+L]
        # k*i > m  =>  k > m/i  =>  k >= floor(m/i) + 1
        # k*i <= m+L => k <= (m+L)/i
        start_k = (m // i) + 1
        end_k = (m + L) // i
        for k in range(start_k, end_k + 1):
            val = k * i
            j = val - m - 1  # offset from m, 0-indexed (val = m + 1 + j)
            if 0 <= j < L:
                adj[i - 1].append(j)
        
        # Early termination: if any index has no candidates, matching is impossible
        if not adj[i - 1]:
            return False

    # Kuhn's algorithm (DFS augmenting paths)
    match_r = [-1] * L  # match_r[j] = i if right node j is matched to left node i
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False

    for u in range(n):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least L > 0 such that matching exists in (m, m+L]."""
    # Lower bound: for each i, the first multiple of i > m is at distance dist_i.
    # L must be at least max(dist_i) because b_i must be in (m, m+L] and divisible by i.
    min_L = 0
    for i in range(1, 13):
        next_mult = ((m // i) + 1) * i
        dist = next_mult - m
        if dist > min_L:
            min_L = dist
    
    # L must be at least 1
    if min_L < 1:
        min_L = 1
        
    for L in range(min_L, 145):
        if _has_matching(m, L):
            return L
    return 144 # Should not happen per contract, but safe fallback

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 12
    max_m = 1000000
    period = 27720  # LCM(1..12)
    
    best_m = 0
    best_f = 0
    
    # We know f(12, m) is periodic with period 27720?
    # Not exactly, because the absolute values matter for divisibility?
    # No, divisibility by i depends only on m mod i.
    # The structure of candidates in (m, m+L] depends on m mod i for each i.
    # Since the condition "i | x" depends on x mod i, and x = m + 1 + j,
    # the pattern of divisibility repeats every LCM(1..12) in m.
    # Therefore, f(12, m) is periodic with period 27720.
    # We only need to search m in [0, 27720).
    
    # Let's verify this logic.
    # If m' = m + 27720, then for any x in (m, m+L], x' = x + 27720 is in (m', m'+L].
    # i | x iff i | (x + 27720) because i | 27720.
    # So the set of valid assignments is isomorphic.
    # Thus f(12, m) = f(12, m + 27720).
    
    # So we only need to search one period.
    # This reduces the search space from 1,000,000 to 27,720.
    # This is a massive speedup.
    
    # Let's scan the entire period.
    # 27720 iterations.
    # Each iteration does binary search or linear search on L (up to 144).
    # Each _has_matching call is O(n * L * avg_degree) roughly, which is small.
    # 27720 * 144 * small_const should be well under 1 second.
    
    # Let's do a full scan of the period.
    for m in range(period):
        if time.time() > deadline:
            break
            
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    # If we finished the scan, we have the global optimum within the period.
    # Since f is periodic, this is the global optimum for all m.
    # We can return best_m directly.
    
    # Just in case we didn't finish (unlikely), we might want to check a few random spots?
    # No, the scan is fast.
    
    return int(best_m)
# EVOLVE-BLOCK-END
