import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[int(a), int(b)] for a in A for b in B]
    
    if _verify_fast(edges, n):
        return edges
    
    # fallback: star
    return [[int(0), int(v)] for v in range(1, n)]

def _bfs_diameter(adj, n):
    """Compute diameter using BFS from each vertex. Returns None if disconnected."""
    maxd = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if -1 in dist:
            return None
        maxd = max(maxd, max(dist))
    return maxd

def _verify_fast(edges, n):
    """Verify diameter == 2 and every edge is critical."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter == 2
    d = _bfs_diameter(adj, n)
    if d != 2:
        return False
    
    # Check each edge is critical
    for u, v in edges:
        # Temporarily remove edge (u,v)
        adj[u].discard(v)
        adj[v].discard(u)
        
        d2 = _bfs_diameter(adj, n)
        if d2 is not None and d2 <= 2:
            # Edge is not critical
            adj[u].add(v)
            adj[v].add(u)
            return False
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
    
    return True
