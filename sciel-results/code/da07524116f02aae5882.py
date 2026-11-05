import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    if k == 5 and g == 9:
        best_n = float('inf')
        best_edges = None
        
        # Try LUW construction with various d values
        for d in [4, 5, 6, 7, 8]:
            if time.time() > deadline - 5:
                break
            n, edges = _construct_luw(k, d)
            if n < best_n and _verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break  # LUW graphs grow with d, so smaller d gives smaller n if valid
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback: try larger d
        n, edges = _construct_luw(k, 8)
        return n, edges
    
    # General case
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw(q, d)

def _construct_luw(q, d):
    """
    Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
    if q == 4:
        add_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        sub_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    else:
        add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
        mul_table = [[x * y % q for y in range(q)] for x in range(q)]
        sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        side, x_tuple = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_table[x_tuple[c]][mul_table[y[a]][x_tuple[b]]]
                else:
                    y[c] = sub_table[x_tuple[c]][mul_table[x_tuple[a]][y[b]]]
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((i, index[other]))
                
    return len(order), edges

def _verify_girth(n, edges, g):
    """
    Verify that the graph has girth at least g.
    Returns True if valid, False otherwise.
    """
    if g <= 1:
        return True
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length 3 to g-1 using non-backtracking walks
    # For each vertex, do a BFS of non-backtracking walks up to depth g-1
    # If we reach the start vertex at depth L where 3 <= L < g, we have a short cycle
    
    for start in range(n):
        # visited[state] = set of depths reached, where state encodes (current, prev)
        visited = {}
        queue = deque()
        
        # Start: at vertex 'start', no previous vertex (use -1)
        queue.append((start, -1, 0))
        visited[(start, -1)] = {0}
        
        while queue:
            u, prev, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            for v in adj[u]:
                if v == prev:
                    continue  # Non-backtracking
                
                new_depth = depth + 1
                
                # Check if we've formed a cycle
                if v == start and new_depth >= 3:
                    if new_depth < g:
                        return False
                    continue
                
                state = (v, u)
                if state in visited:
                    if new_depth in visited[state]:
                        continue
                    else:
                        visited[state].add(new_depth)
                else:
                    visited[state] = {new_depth}
                
                if new_depth < g - 1:
                    queue.append((v, u, new_depth))
    
    return True
