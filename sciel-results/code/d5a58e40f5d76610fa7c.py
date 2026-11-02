import time
import sys
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for guaranteed validity.
    For k=3, g=16, this yields n=4374.
    """
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin for 120s limit

    # For k=3, the LUW construction CD(d, 3) has girth 2d+2.
    # We need girth >= 16, so 2d+2 >= 16 => d >= 7.
    # n = 2 * 3^d.
    # For d=7, n = 2 * 2187 = 4374.
    
    # We will construct LUW(7, 3).
    d = (g + 1) // 2  # d=7 for g=16
    q = k             # q=3
    
    try:
        n, edges = _luw_construct_fast(q, d)
        # Verify girth to be safe
        # If girth is insufficient, increase d
        girth = _calc_girth_fast(n, edges)
        while girth < g:
            d += 1
            n, edges = _luw_construct_fast(q, d)
            girth = _calc_girth_fast(n, edges)
            if time.time() > deadline:
                break
                
        return n, edges
    except Exception:
        # Fallback to a simple cycle or random if LUW fails (unlikely)
        # But LUW is deterministic and should work.
        pass

    # If all else fails, return a trivial graph (will score 0 but avoid crash)
    return 0, []

def _luw_construct_fast(q, d):
    """
    Fast LUW construction.
    """
    if q == 4:
        # Not needed for k=3, but keep for generality
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Precompute rules
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Generate vertices
    # Vertices are (side, tuple of length d+1)
    # Side 0: (0, (None, x1, ..., xd))
    # Side 1: (1, (None, y1, ..., yd))
    
    # We can map tuples to integers for faster lookup
    # Total vertices = 2 * q^d
    # Index = side * q^d + value
    
    q_pow_d = q ** d
    n = 2 * q_pow_d
    
    # We need to generate the edges.
    # For each vertex on side 0, find its 3 neighbors on side 1.
    # Neighbors are determined by the first coordinate 'first' in range(q).
    
    edges = []
    
    # Helper to convert tuple to int
    def tuple_to_int(t):
        val = 0
        for i in range(1, d + 1):
            val = val * q + t[i]
        return val

    def int_to_tuple(val):
        t = [0] * (d + 1)
        for i in range(d, 0, -1):
            t[i] = val % q
            val //= q
        return tuple(t)

    # Iterate over all vertices on side 0
    for val0 in range(q_pow_d):
        x = int_to_tuple(val0)
        x = (None,) + x
        
        for first in range(q):
            # Calculate neighbor y
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b]))
            
            val1 = tuple_to_int(tuple(y))
            idx_u = val0
            idx_v = q_pow_d + val1
            edges.append((idx_u, idx_v))
            
    return n, edges

def _calc_girth_fast(n, edges):
    """
    Fast girth calculation using BFS.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < 16:
                            return min_girth
        
        if min_girth < 16:
            break
            
    return min_girth if min_girth != float('inf') else n

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    girth = _calc_girth_fast(n, edges)
    print(f"Measured girth: {girth}")
