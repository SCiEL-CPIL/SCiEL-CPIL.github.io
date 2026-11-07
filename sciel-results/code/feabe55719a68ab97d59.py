import time

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # Build adjacency: for each i (0-indexed), list of candidate values
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        # First multiple of i > m
        first = ((m // i) + 1) * i
        end = m + L
        v = first
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # If any i has no candidates, return False
    for i in range(n):
        if not adj[i]:
            return False
    
    # Kuhn's algorithm for bipartite matching
    # match_val: value -> i (0-indexed)
    match_val = {}
    
    def dfs(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_val or dfs(match_val[v], visited):
                match_val[v] = i
                return True
        return False
    
    for i in range(n):
        visited = set()
        if not dfs(i, visited):
            return False
    return True

def _f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L
    lo, hi = 1, 144  # f is at most 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 12
    period = 27720  # lcm(1, 2, ..., 12)
    
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
