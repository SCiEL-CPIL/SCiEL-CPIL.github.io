"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
MAX_START = 1000000

def _can_match(m, L):
    """Check if (m, m+L] contains distinct b_1..b_N with i | b_i."""
    n = N
    # Build adjacency: for each i (1-indexed), list of candidate values in (m, m+L]
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
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
    # Left side: indices 0..n-1 (representing i=1..n)
    # Right side: values (multiples)
    match_val = {}  # value -> left index assigned
    
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

def _f_val(m):
    """Compute f(N, m): least positive L such that (m, m+L] has distinct b_1..b_N with i|b_i."""
    # Lower bound: at least N (need N distinct values)
    # Upper bound: try increasing L
    # The maximum L is bounded; for N=12, L is at most ~144 per task description
    lo = 1
    hi = 200  # safe upper bound
    
    # First check if hi is feasible
    if not _can_match(m, hi):
        # Try larger
        hi = 500
        if not _can_match(m, hi):
            hi = 1000
            if not _can_match(m, hi):
                return 0  # shouldn't happen
    
    # Binary search for minimum L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    """Find m in [0, MAX_START] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s budget
    
    best_m = 0
    best_f = 0
    
    # Pre-compute for a set of candidate m values
    # Strategy: sample densely, then refine around promising regions
    
    # First pass: coarse sampling
    step = 100
    candidates = list(range(0, MAX_START + 1, step))
    
    # Evaluate candidates
    for m in candidates:
        if time.time() > deadline:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # Second pass: fine sampling around the best region
    if best_f > 0:
        # Look in a window around best_m
        window = 500
        lo_m = max(0, best_m - window)
        hi_m = min(MAX_START, best_m + window)
        for m in range(lo_m, hi_m + 1):
            if time.time() > deadline:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    # Third pass: even finer around the best
    if best_f > 0:
        window = 50
        lo_m = max(0, best_m - window)
        hi_m = min(MAX_START, best_m + window)
        for m in range(lo_m, hi_m + 1):
            if time.time() > deadline:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    # Final check: also try some specific interesting values
    special_ms = [0, 1, 2, 11, 12, 13, 60, 120, 252, 504, 1008, 
                  2520, 5040, 10080, 25200, 50400, 100800, 252000, 504000, 1000000]
    for m in special_ms:
        if time.time() > deadline:
            break
        if 0 <= m <= MAX_START:
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
