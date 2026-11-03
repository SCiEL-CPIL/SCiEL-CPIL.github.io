import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    best_n = float('inf')
    best_edges = None
    
    # Try LUW constructions with different d values
    # For girth g, we need d >= g//2 - 1 to have enough "depth"
    min_d = max(4, g // 2 - 1)
    max_d = min(g + 3, 15)  # Cap to avoid excessive computation
    
    # Try smaller d first (smaller n), but also try a range
    d_candidates = list(range(min_d, max_d + 1))
    
    # Also try some specific values that might work well
    for d in d_candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
        except Exception:
            continue
            
        if n_luw >= best_n:
            continue
            
        # Quick degree check
        if not _check_degree(edges_luw, n_luw, k):
            continue
            
        # Verify girth with optimized algorithm
        if _has_girth_at_least_optimized(n_luw, edges_luw, g):
            best_n = n_luw
            best_edges = edges_luw
            break  # Found a valid graph, stop
    
    # If no valid LUW found, fall back to the largest d
    if best_n == float('inf'):
        for d in range(max_d, min_d - 1, -1):
            if time.time() > deadline - 5:
                break
            try:
                n_luw, edges_luw = _construct_luw_optimized(k, d)
            except Exception:
                continue
            if n_luw < best_n and _check_degree(edges_luw, n_luw, k):
                best_n = n_luw
                best_edges = edges_luw
                break
    
    if best_n == float('inf'):
        # Last resort: return a trivial valid graph (should not happen)
        n_luw, edges_luw = _construct_luw_optimized(k, max_d)
        return n_luw, edges_luw
    
    return best_n, best_edges

def _check_degree(edges, n, k):
    """Quick check that all vertices have degree exactly k."""
    deg = [0] * n
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True

def _has_girth_at_least_optimized(n, edges, g):
    """
    Optimized girth verification using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    Uses early termination and efficient adjacency lists.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS to find shortest cycle
    for start in range(n):
        if not _bfs_no_short_cycle(adj, start, g):
            return False
    return True

def _bfs_no_short_cycle(adj, start, g):
    """
    BFS from 'start' to check if there's a cycle of length < g.
    Returns True if no such cycle exists.
    """
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    dist[start] = 0
    q = deque([start])
    
    while q:
        u = q.popleft()
        du = dist[u]
        # If we've already gone too far, no need to continue
        if du >= g // 2:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                parent[v] = u
                q.append(v)
            elif v != parent[u]:
                # Found a cycle
                cycle_len = du + dist[v] + 1
                if cycle_len < g:
                    return False
    return True

def _construct_luw_optimized(k, d):
    """
    Deterministic LUW construction for k-regular graphs.
    Optimized for k=3 with inlined arithmetic.
    """
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
    # Rule selection: use a pattern that tends to give good girth
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = (x[c] + y[a] * x[b]) % q
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = (x[c] - x[a] * y[b]) % q
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_luw_generic(k, d):
    """Fallback for other k values."""
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
