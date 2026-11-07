import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    n = 12
    period = 27720  # lcm(1,2,...,12)
    
    def can_match(m, L):
        # Build adjacency: for each i (1..n), offsets j in 1..L where (m+j) % i == 0
        adj = [None] * (n + 1)
        for i in range(1, n + 1):
            r = (-m) % i
            first = i if r == 0 else r
            if first > L:
                return False
            adj[i] = list(range(first, L + 1, i))
        
        # Order by scarcity for faster matching
        order = sorted(range(1, n + 1), key=lambda i: len(adj[i]))
        
        # Bipartite matching via DFS augmenting paths
        match_to_val = {}
        
        def dfs(i, visited):
            for v in adj[i]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_to_val or dfs(match_to_val[v], visited):
                    match_to_val[v] = i
                    return True
            return False
        
        for i in order:
            visited = set()
            if not dfs(i, visited):
                return False
        
        return True
    
    def f_of_m(m):
        # Binary search on L: find least L >= 1 such that can_match(m, L)
        lo, hi = 1, 144
        if not can_match(m, hi):
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Exploit periodicity: f(12, m) depends only on m mod 27720
    # Search all residues in one period to find the true maximum
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # best_m is in [0, 27719] which is within [0, 1000000]
    return int(best_m)
