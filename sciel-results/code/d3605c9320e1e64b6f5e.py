import time
import sys

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using DFS matching."""
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
    """Binary search smallest L in [1, 144] such that matching exists."""
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

def _f_fast(m, n=12):
    """Optimized: check from high L down to find the minimum quickly for common cases."""
    # First check if L=144 works (should always)
    # Binary search is fine for n=12
    return _f(m, n)

def construct():
    deadline = time.time() + 85  # 85s budget, 5s safety margin
    
    best_m = 12
    best_f = _f(12)
    
    # Phase 1: Coarse scan with step 5 for better coverage
    step = 5
    m = 0
    while m <= 1000000:
        if time.time() > deadline:
            break
        f = _f(m)
        if f > best_f:
            best_f = f
            best_m = m
        m += step
    
    # Phase 2: Refine around best_m with step 1 in a wider range
    if time.time() < deadline:
        for dm in range(-300, 301):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    # Phase 3: Tighter refinement
    if time.time() < deadline:
        for dm in range(-30, 31):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    # Phase 4: If time remains, do a second coarse pass with different step
    if time.time() < deadline - 10:
        step2 = 11
        m = 0
        while m <= 1000000:
            if time.time() > deadline:
                break
            f = _f(m)
            if f > best_f:
                best_f = f
                best_m = m
            m += step2
        
        # Refine again around new best
        if time.time() < deadline:
            for dm in range(-100, 101):
                if time.time() > deadline:
                    break
                mm = best_m + dm
                if mm < 0 or mm > 1000000:
                    continue
                f = _f(mm)
                if f > best_f:
                    best_f = f
                    best_m = mm
    
    return int(best_m)
