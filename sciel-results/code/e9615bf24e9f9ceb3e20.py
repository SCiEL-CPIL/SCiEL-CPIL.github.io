import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Keep margin for overhead (90s kill)
    
    N = 12
    LCM12 = 27720  # lcm(1,2,...,12)
    MAX_L = 200
    
    def check_match(m, L):
        """Check if (m, m+L] contains distinct b_1..b_N with i | b_i."""
        # Build adjacency: for each i in 1..N, list of multiples of i in (m, m+L]
        adj = [[] for _ in range(N)]
        for i in range(1, N + 1):
            # First multiple of i > m
            first = ((m // i) + 1) * i
            end = m + L
            v = first
            while v <= end:
                adj[i - 1].append(v)
                v += i
        
        # If any i has no candidates, fail
        for i in range(N):
            if not adj[i]:
                return False
        
        # Kuhn's algorithm for bipartite matching
        # match_val maps value -> i (0-indexed)
        match_val = {}
        
        def dfs(i, visited):
            for v in adj[i]:
                if v in visited:
                    continue
                visited.add(v)
                # If v is unmatched or the previous match can be reassigned
                if v not in match_val or dfs(match_val[v], visited):
                    match_val[v] = i
                    return True
            return False
        
        for i in range(N):
            visited = set()
            if not dfs(i, visited):
                return False
        
        return True
    
    def f(m, n=12):
        """Find the least positive L such that (m, m+L] contains distinct b_1..b_n with i | b_i."""
        # Binary search on L
        lo, hi = 1, MAX_L
        # First check if L=hi works
        if not check_match(m, hi):
            # Try larger
            hi = 400
            if not check_match(m, hi):
                hi = 800
                if not check_match(m, hi):
                    return 0
        
        while lo < hi:
            mid = (lo + hi) // 2
            if check_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_L = f(0)
    
    # Phase 1: Coarse search over m in [0, 1000000]
    # Use a step that balances coverage and time
    # Each f(m) call does ~log2(200) ≈ 8 check_match calls
    # Each check_match is O(N * L / N) ≈ O(L) for building adj + O(N^2 * L/N) for matching
    # So each f(m) is roughly O(N * log(MAX_L) * L) ≈ 12 * 8 * 200 = 19200 ops
    # In 80 seconds, we can do roughly 80 / (19200/1e6) ≈ 4000 evaluations (very rough)
    # Let's be conservative and use a step of 50, giving 20000 candidates
    # But we need to check time frequently
    
    # Try to cover the range with a reasonable step
    # Start with step 200 (5000 candidates)
    step = 200
    for m in range(0, 1000001, step):
        if time.time() > deadline - 10:
            break
        L_val = f(m)
        if L_val > best_L:
            best_L = L_val
            best_m = m
    
    # Phase 2: Fine search around the best m found
    if time.time() < deadline - 5:
        search_range = 500
        for m in range(max(0, best_m - search_range), min(1000001, best_m + search_range)):
            if time.time() > deadline - 3:
                break
            L_val = f(m)
            if L_val > best_L:
                best_L = L_val
                best_m = m
    
    # Phase 3: Targeted search near multiples of LCM12 - 1
    # These are likely to be "worst" starts because just before a multiple of all divisors
    if time.time() < deadline - 3:
        for k in range(1, 40):
            m = k * LCM12 - 1
            if m < 0 or m > 1000000:
                continue
            if time.time() > deadline - 2:
                break
            L_val = f(m)
            if L_val > best_L:
                best_L = L_val
                best_m = m
    
    # Phase 4: If we have more time, do another coarse pass with smaller step
    if time.time() < deadline - 5:
        step2 = 50
        for m in range(0, 1000001, step2):
            if time.time() > deadline - 2:
                break
            L_val = f(m)
            if L_val > best_L:
                best_L = L_val
                best_m = m
    
    return int(best_m)
