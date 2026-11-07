import sys
sys.setrecursionlimit(10000)

def _precompute():
    n = 12
    period = 27720
    max_L = 144
    
    # For each residue m mod period, find minimal L such that
    # there exist distinct b_1,...,b_n in (m, m+L] with i | b_i
    # i.e., for each i, there exists d_i in 1..L with (m + d_i) % i == 0,
    # and all d_i are distinct.
    
    results = [0] * period
    
    for m in range(period):
        found_L = 0
        for L in range(1, max_L + 1):
            # Build adjacency: for each i (1..n), valid d in 1..L where (m+d) % i == 0
            adj = [[] for _ in range(n)]
            for i in range(1, n + 1):
                # d ≡ -m (mod i), d in 1..L
                start_d = (-m) % i
                if start_d == 0:
                    start_d = i
                d = start_d
                while d <= L:
                    adj[i - 1].append(d)
                    d += i
            
            # Check if perfect matching exists using DFS augmenting paths
            # Left side: i in 0..n-1, Right side: d in 1..L
            match = [-1] * (L + 1)  # match[d] = i_idx or -1
            
            def dfs(i_idx, visited):
                for d in adj[i_idx]:
                    if visited[d]:
                        continue
                    visited[d] = True
                    if match[d] == -1 or dfs(match[d], visited):
                        match[d] = i_idx
                        return True
                return False
            
            ok = True
            for i_idx in range(n):
                visited = [False] * (L + 1)
                if not dfs(i_idx, visited):
                    ok = False
                    break
            
            if ok:
                found_L = L
                break
        
        results[m] = found_L if found_L > 0 else max_L
    
    return results

_results = _precompute()

def construct():
    period = 27720
    best_r = 0
    best_val = -1
    for r in range(period):
        if _results[r] > best_val:
            best_val = _results[r]
            best_r = r
    return best_r
