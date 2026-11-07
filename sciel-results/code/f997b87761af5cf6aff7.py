import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 12
    # LCM(1..12) = 27720
    PERIOD = 27720
    
    def has_matching(m, L):
        """Check if there's a perfect matching in the bipartite graph.
        Left: i=1..12, Right: values v in (m, m+L]
        Edge (i,v) exists if i divides v and all b_i are distinct.
        """
        if L < N:
            return False
            
        # Build adjacency: for each i, which right-node indices can be matched
        # Right node j corresponds to value v = m + 1 + j
        adj = [[] for _ in range(N)]
        
        for i in range(1, N + 1):
            # Find multiples of i in (m, m+L]
            first = ((m // i) + 1) * i
            if first > m + L:
                continue
            for v in range(first, m + L + 1, i):
                idx = v - (m + 1)
                adj[i - 1].append(idx)
                
        # Hopcroft-Karp algorithm for bipartite matching
        # matchL[i] = right node matched to left node i
        # matchR[j] = left node matched to right node j
        matchL = [-1] * N
        matchR = [-1] * L
        dist = [0] * N
        
        def bfs():
            """Find augmenting paths of shortest length."""
            queue = deque()
            for i in range(N):
                if matchL[i] == -1:
                    dist[i] = 0
                    queue.append(i)
                else:
                    dist[i] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
            
        def dfs(u):
            """Try to find an augmenting path from left node u."""
            for v in adj[u]:
                mu = matchR[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False
            
        # Main Hopcroft-Karp loop
        matching = 0
        while bfs():
            for i in range(N):
                if matchL[i] == -1:
                    if dfs(i):
                        matching += 1
        return matching == N

    best_L = 0
    best_m = 0
    
    # Search over all m in one period since the problem is periodic with period 27720
    for m in range(PERIOD):
        # Check time
        if time.time() > deadline:
            break
            
        # Find minimal L for this m using binary search
        # f(12,m) is at most 144, so search in [1, 144]
        lo, hi = 1, 144
        found_L = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                found_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if found_L is not None:
            if found_L > best_L:
                best_L = found_L
                best_m = m
    
    return best_m
