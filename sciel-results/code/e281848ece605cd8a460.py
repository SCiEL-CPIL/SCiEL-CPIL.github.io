"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}

def _check_all_pairs_bfs_edges(edges, n):
    """Check that for every edge, its removal increases diameter above 2 or disconnects."""
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges_set and (j, i) not in edges_set:
                # Check diameter of full graph
                pass
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check diameter is exactly 2
    for s in range(n):
        # BFS
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for t in range(n):
            if t == s:
                continue
            if dist[t] == -1:
                return False
            if dist[t] > 2:
                return False
    
    # Check every edge is critical
    for idx, (u, v) in enumerate(edges):
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                x = q.popleft()
                for y in adj2[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        q.append(y)
            for t in range(n):
                if t == s:
                    continue
                if dist[t] == -1 or dist[t] > 2:
                    # This is fine - diameter increased or disconnected
                    break
            else:
                # All pairs reachable within 2 - this edge is NOT critical
                return False
    return True

def construct():
    """Return edges of K(8,8) on vertices 0..15."""
    n = P['n']
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    
    # Verify types are correct
    result = []
    for e in edges:
        u, v = e
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
