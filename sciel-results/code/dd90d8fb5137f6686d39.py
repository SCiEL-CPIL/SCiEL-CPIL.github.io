import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = 12
    
    def can_match(m, L):
        # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
        # Build candidate sets for each i
        # For i, candidates are multiples of i in (m, m+L]
        # We need to assign distinct values to each i
        
        # For each i from 1 to n, get list of multiples of i in (m, m+L]
        # Then check if there's a system of distinct representatives
        
        # Since n=12 and L<=144, the window is small
        # Use bipartite matching (Hungarian or DFS-based)
        
        # Collect all candidate values in (m, m+L]
        # For each i, which values are divisible by i?
        
        # Build adjacency: for each i (1..n), list of candidate values divisible by i
        # Then do bipartite matching
        
        # Get all distinct candidate values
        lo = m + 1
        hi = m + L
        candidates = list(range(lo, hi + 1))
        
        # For each i, filter candidates divisible by i
        adj = [None] * (n + 1)  # 1-indexed
        for i in range(1, n + 1):
            adj[i] = [v for v in candidates if v % i == 0]
        
        # Bipartite matching: left = {1..n}, right = candidates
        # Use DFS-based augmenting path
        match_to_val = {}  # val -> i (which i is matched to this val)
        
        def dfs(i, visited):
            for v in adj[i]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_to_val or dfs(match_to_val[v], visited):
                    match_to_val[v] = i
                    return True
            return False
        
        count = 0
        for i in range(1, n + 1):
            visited = set()
            if dfs(i, visited):
                count += 1
        
        return count == n
    
    def f_of_m(m):
        # Binary search on L: find least L >= 1 such that can_match(m, L)
        # L is at most 144
        lo, hi = 1, 144
        if not can_match(m, hi):
            # Shouldn't happen per problem statement, but handle
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Search over m values
    # Strategy: test m values that are likely to be "bad"
    # m values just after multiples of small numbers create gaps
    # But we need to search broadly
    
    best_m = 12
    best_f = f_of_m(12)
    
    # First pass: test a grid of m values
    # The function f(12,m) is periodic-ish with period lcm(1..12) = 27720
    # So we can sample one period and find the max, then verify
    
    # lcm(1,2,...,12) = 27720
    period = 27720
    
    # Sample m values: every k-th value in the period, plus targeted spots
    # Start with coarse sampling
    step = 100
    m = 0
    while m <= 1000000:
        if time.time() > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Refine around the best found
    if time.time() < deadline - 5:
        # Fine-grained search around best_m
        for dm in range(-200, 201):
            if time.time() > deadline:
                break
            m_candidate = best_m + dm
            if 0 <= m_candidate <= 1000000:
                f_val = f_of_m(m_candidate)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_candidate
    
    # Also check specific "interesting" m values
    if time.time() < deadline - 2:
        # Check m values that are 1 less than multiples of small numbers
        # These create gaps
        for base in range(0, 1000001, 500):
            if time.time() > deadline:
                break
            for i in range(1, 13):
                # m = base - 1 (just before a multiple region)
                m_candidate = base - 1
                if 0 <= m_candidate <= 1000000:
                    f_val = f_of_m(m_candidate)
                    if f_val > best_f:
                        best_f = f_val
                        best_m = m_candidate
    
    return int(best_m)
