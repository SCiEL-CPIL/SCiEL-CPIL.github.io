import time
import sys
import random

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using bipartite matching."""
    # Build adjacency: for each i in 1..n, list of multiples of i in (m, m+L]
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        end = m + L
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # Bipartite matching via DFS augmenting paths
    match_r = {}
    
    sys.setrecursionlimit(10000)
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_r or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n):
        visited = set()
        if not dfs(u, visited):
            return False
    return True

def _f(m, n=12):
    """Find smallest L in [1, 144] such that (m, m+L] has valid matching."""
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 85  # 85s budget, 5s safety margin for return
    
    # Key insight: f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to evaluate m in [0, 27719] and pick the best residue
    period = 27720
    
    best_residue = 0
    best_f = 0
    
    # Exhaustive search over one period
    for r in range(period):
        if time.time() > deadline:
            break
        f = _f(r)
        if f > best_f:
            best_f = f
            best_residue = r
    
    # Find the largest m <= 1000000 with m % period == best_residue
    # m = best_residue + k * period <= 1000000
    # k_max = (1000000 - best_residue) // period
    if best_residue <= 1000000:
        k_max = (1000000 - best_residue) // period
        best_m = best_residue + k_max * period
    else:
        best_m = 0
    
    # Verify and potentially improve with nearby values in case of edge effects
    # Check a few values around best_m for robustness
    if time.time() < deadline - 5:
        candidates = set()
        candidates.add(best_m)
        for delta in range(-3, 4):
            mm = best_m + delta
            if 0 <= mm <= 1000000:
                candidates.add(mm)
        
        for mm in candidates:
            if time.time() > deadline - 5:
                break
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    return int(best_m)
