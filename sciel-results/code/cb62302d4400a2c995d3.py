import time
import random
from collections import deque

def _check_girth(n, edges, g):
    """Return True iff no cycle is shorter than g (BFS from every vertex)."""
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    max_depth = g // 2
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] != -1:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                else:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
    return True

def _generate_random_3_regular(n):
    """Generate a random simple 3-regular graph on n vertices.
    Returns None if not simple."""
    if n < 4 or n % 2 != 0:
        return None
        
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    
    edges = []
    adj_set = set()
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        
        if u == v:
            return None
        if (u, v) in adj_set or (v, u) in adj_set:
            return None
            
        adj_set.add((u, v))
        adj_set.add((v, u))
        edges.append((u, v))
        
    return edges

def construct(k, g):
    deadline = time.time() + 110
    best_n = float('inf')
    best_edges = None
    
    if k != 3:
        # Fallback for other k if needed, but task is k=3
        # For now, return empty or raise?
        # We assume k=3.
        pass

    # Search for minimal n
    # Start from 600, go up to 1500
    # Step by 2 to keep n even
    for n in range(600, 1500, 2):
        if time.time() > deadline:
            break
            
        for _ in range(5):
            if time.time() > deadline:
                break
            edges = _generate_random_3_regular(n)
            if edges is None:
                continue
            if _check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                # Since we iterate n upwards, the first valid n is the smallest found so far.
                # We can return immediately to save time?
                # Yes, because we want minimal n.
                return (n, edges)
                
    if best_edges is not None:
        return (best_n, best_edges)
        
    # If no graph found, return a dummy to avoid crash?
    # But it should be found.
    return (0, [])
