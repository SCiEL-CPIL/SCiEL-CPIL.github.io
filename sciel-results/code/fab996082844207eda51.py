import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3:
        # Fallback to LUW for other k
        return _construct_luw_fallback(k, g)

    # Strategy: Randomized Local Search
    # 1. Determine lower bound n_min
    # Moore bound for k=3, g=16 (m=8): 2 * (2^8 - 1) = 510.
    # Prompt says lower bound 512.
    # We will try n from 512 upwards.
    
    n_min = 512
    # Step size: 2 (since n must be even for 3-regular graph)
    
    # Try to find a graph with n = n_min, n_min+2, ...
    # Limit the search range to avoid timeout
    # If n is too large, LUW might be better?
    # LUW for d=8 (girth 16) yields n = ?
    # Let's stick to random search up to n=1000.
    
    best_n = float('inf')
    best_edges = None
    
    # Try random local search
    for n in range(n_min, 1000, 2):
        if time.time() > deadline - 20:
            break
            
        # Try multiple random restarts
        found = False
        for restart in range(20):
            if time.time() > deadline - 20:
                break
                
            # Generate random 3-regular graph
            edges = _generate_random_3_regular(n)
            if edges is None:
                continue
                
            # Check girth
            if _has_girth_at_least_fast(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
            
            # Local search: try to increase girth by swapping edges
            # This is complex. Instead, just try more random graphs.
            # Or, use a "repair" strategy.
            
        if found:
            break
            
    if best_n != float('inf'):
        return best_n, best_edges

    # Fallback to LUW if random search fails
    # The current LUW implementation in the seed might be slow or yield large n.
    # We will use the optimized LUW from the current program but ensure it runs.
    # The current program's LUW is in _construct_luw_optimized.
    # We should call it.
    
    # Re-implement or call existing
    # The existing code has _construct_luw_optimized.
    # Let's try d=8, 9, 10.
    
    for d in range(8, 15):
        if time.time() > deadline - 5:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(3, d)
            if n_luw < best_n and _has_girth_at_least_fast(n_luw, edges_luw, g):
                best_n = n_luw
                best_edges = edges_luw
        except Exception:
            continue
            
    if best_n == float('inf'):
        # Last resort: return a large valid graph
        # Use a very large d for LUW
        n_luw, edges_luw = _construct_luw_optimized(3, 20)
        return n_luw, edges_luw

    return best_n, best_edges

def _generate_random_3_regular(n):
    # Generate a random 3-regular graph
    # Using configuration model
    # List of half-edges
    half_edges = []
    for i in range(n):
        half_edges.extend([i, i, i])
        
    random.shuffle(half_edges)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, 2*n, 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return list(edges)

def _has_girth_at_least_fast(n, edges, g):
    # Optimized girth check
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    # Optimization: Stop if we find a cycle < g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_fallback(k, g):
    # Reuse the logic from the current program's LUW
    # We need to include the helper functions from the current program
    # Since I am replacing the block, I must include them or assume they exist?
    # The prompt says "Keep the function names and signatures of the seed program".
    # The current program HAS _construct_luw_optimized.
    # I should call it.
    
    min_d = max(3, g // 2 - 2)
    for d in range(min_d, min_d + 10):
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    return _construct_luw_optimized(k, min_d + 10)

def _construct_luw_optimized(k, d):
    """
    Deterministic LUW construction for k-regular graphs.
    Optimized for k=3.
    """
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
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

def _construct_luw_generic(k, d):
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
