import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85  # 85 seconds to be safe
    
    N = 12
    LCM = 27720  # lcm(1,2,...,12)
    
    def can_match(m, L):
        """Check if there's a valid matching for start m and length L."""
        # Candidates for each position i: multiples of i in (m, m+L]
        # Build bipartite graph and check perfect matching
        # Left: positions 1..12, Right: candidate numbers
        
        # For each position i, find multiples of i in (m, m+L]
        # First multiple of i > m: floor(m/i)*i + i if m%i != 0, else m+i... 
        # Actually: the smallest multiple of i > m is (m // i + 1) * i
        # The largest multiple of i <= m+L is ((m+L) // i) * i
        
        adj = []
        for i in range(1, N+1):
            first = (m // i + 1) * i
            last = ((m + L) // i) * i
            if first > last:
                adj.append([])
            else:
                # Multiples: first, first+i, first+2i, ..., last
                adj.append(list(range(first, last + 1, i)))
        
        # Now do bipartite matching
        # Right side: all candidate numbers that appear in any adj[i]
        # Use Hungarian or DFS-based matching (Hopcroft-Karp or simple DFS)
        
        # Since N=12 is small, use DFS-based matching
        # Map right side to indices
        all_candidates = set()
        for i in range(N):
            for x in adj[i]:
                all_candidates.add(x)
        
        cand_list = list(all_candidates)
        cand_to_idx = {x: idx for idx, x in enumerate(cand_list)}
        
        # Build adjacency in terms of indices
        adj_idx = []
        for i in range(N):
            adj_idx.append([cand_to_idx[x] for x in adj[i]])
        
        # DFS-based bipartite matching
        match_to_cand = [-1] * len(cand_list)  # which position is matched to which candidate
        
        def dfs(pos, visited):
            for c in adj_idx[pos]:
                if visited[c]:
                    continue
                visited[c] = True
                if match_to_cand[c] == -1 or dfs(match_to_cand[c], visited):
                    match_to_cand[c] = pos
                    return True
            return False
        
        matched = 0
        for pos in range(N):
            visited = [False] * len(cand_list)
            if dfs(pos, visited):
                matched += 1
        
        return matched == N
    
    def f_12(m):
        """Find minimum L such that can_match(m, L) is True."""
        # Binary search on L from 1 to 144
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Compute f_12(r) for all r in 0..LCM-1
    best_m = 0
    best_L = 0
    
    for r in range(LCM):
        if time.time() - start_time > deadline:
            break
        L_val = f_12(r)
        if L_val > best_L:
            best_L = L_val
            best_m = r
    
    # Return best_m (which is in 0..27719, so within 0..1000000)
    return int(best_m)
