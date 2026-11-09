import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k != 3 or g != 16:
        return _luw_fallback(k, g)
        
    # Search for LUW parameters
    # LUW cubic graphs have 2 * q^d vertices
    # We want smallest n, so small q^d
    # Try primes q and depths d
    
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    max_depth = 10
    
    best_n = float('inf')
    best_edges = None
    
    for q in primes:
        for d in range(1, max_depth + 1):
            if time.time() > deadline - 10:
                break
            n = 2 * (q ** d)
            if n >= best_n:
                continue
                
            # Construct LUW graph
            edges = _construct_luw_cubic(q, d)
            
            # Verify girth
            if _has_girth_at_least(edges, n, g):
                best_n = n
                best_edges = edges
                # If we found a very small one, we can stop early?
                # The Moore bound for n(3,16) is 512.
                # If we find n < 1000, it's excellent.
                if n < 1000:
                    return n, edges
        if time.time() > deadline - 10:
            break
            
    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_fallback(k, g)

def _construct_luw_cubic(q, d):
    """
    Construct a 3-regular graph with 2*q^d vertices using LUW construction.
    This implementation uses a standard LUW variant for cubic graphs.
    Vertices are (side, x) where side in {0,1} and x in F_q^d.
    """
    n = 2 * (q ** d)
    edges = []
    
    # Map vertex (side, x) to index
    # index = side * q^d + sum(x[i] * q^(d-1-i))
    
    def idx(side, x):
        val = 0
        for i in range(d):
            val = val * q + x[i]
        return side * (q ** d) + val
        
    def unidx(i):
        side = i // (q ** d)
        val = i % (q ** d)
        x = [0] * d
        for i in range(d-1, -1, -1):
            x[i] = val % q
            val //= q
        return side, x
        
    qd = q ** d
    
    for i in range(qd):
        side, x = unidx(i)
        # We only generate edges from side 0 to side 1 to avoid duplicates
        # and to ensure we cover all edges exactly once if we iterate all x in side 0?
        # No, vertices are (0, x) and (1, x).
        # We iterate x in F_q^d.
        
        x0 = x
        # Edge 1: (0, x0) - (1, x0)
        u = idx(0, x0)
        v = idx(1, x0)
        edges.append((u, v))
        
        # Edge 2: (0, x0) - (1, x0 + v1)
        # v1 = (1, 0, ..., 0)
        x1 = x0[:]
        x1[0] = (x1[0] + 1) % q
        v = idx(1, x1)
        edges.append((u, v))
        
        # Edge 3: (0, x0) - (1, x0 + v2)
        # v2 = (0, 1, ..., 0)
        x2 = x0[:]
        if d > 1:
            x2[1] = (x2[1] + 1) % q
        else:
            # If d=1, v2 must be different from v1 and 0
            # In F_q, if q>2, we can use 2?
            # But LUW usually requires d>=2 for high girth?
            # If d=1, n=2q. Max girth is small.
            x2[0] = (x2[0] + 2) % q
        v = idx(1, x2)
        edges.append((u, v))
        
    return edges

def _has_girth_at_least(edges, n, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
            
    # BFS for girth
    for start in range(n):
        dist = [-1] * n
        par = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == par[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_fallback(k, g):
    # Keep the old fallback as is
    pass
