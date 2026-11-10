"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i|b_i using bipartite matching."""
    lo = m + 1
    hi = m + L
    
    # For each i (1..n), find valid numbers in (lo, hi]
    # Build adjacency: left nodes are indices 0..n-1 (representing i=1..n)
    # right nodes are the actual values
    valid = [[] for _ in range(n)]
    for i in range(n):
        idx = i + 1
        first = ((lo + idx - 1) // idx) * idx
        if first > hi:
            return False
        x = first
        while x <= hi:
            valid[i].append(x)
            x += idx
    
    # Use DFS-based bipartite matching (augmenting paths)
    # Match right node -> left node
    match_r = {}
    
    def dfs(u, seen):
        for v in valid[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False
    
    count = 0
    for u in range(n):
        seen = set()
        if dfs(u, seen):
            count += 1
    
    return count == n

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search for the minimum L
    low = 1
    high = max_L
    ans = max_L
    
    while low <= high:
        mid = (low + high) // 2
        if _has_matching(m, mid, n):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return ans

def construct():
    """Return m in 0..1000000 that maximizes f(12, m)."""
    deadline = time.time() + 85  # Safety margin (90s limit)
    
    # Period is lcm(1..12) = 27720
    K = 27720
    best_m = 0
    best_L = 0
    
    for r in range(K):
        if time.time() > deadline:
            break
        L = _compute_f(r, n=12, max_L=144)
        if L > best_L:
            best_L = L
            best_m = r
            
    return int(best_m)
# EVOLVE-BLOCK-END
