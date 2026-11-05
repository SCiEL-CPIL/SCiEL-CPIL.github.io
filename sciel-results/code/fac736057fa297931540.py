# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None, t_start=None, deadline=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        if deadline is not None and time.time() > deadline:
            return False
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g, rule_override=None):
    """
    LUW construction with optional rule override.
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    # Apply rule override if provided
    if rule_override:
        for c, val in rule_override.items():
            rule[c] = val

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

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Baseline: LUW
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    if k != 4 or g != 10:
        return best_n, best_edges
    
    # For k=4, g=10, try variations of the rule
    # The rule for c=5 is the key parameter
    base_options = [(1, 3), (4, 1), (5, 1), (1, 6)]
    
    variations = []
    
    # Variation 1: Different rule[5]
    for opt in base_options:
        variations.append({5: opt})
    
    # Variation 2: Different rule[4]
    for opt in [(1, 2), (2, 1), (3, 1), (1, 3)]:
        variations.append({4: opt})
    
    # Variation 3: Different rule[3]
    for opt in [(2, 1), (1, 1), (3, 1), (1, 2)]:
        variations.append({3: opt})
    
    # Variation 4: Combined
    variations.append({4: (2, 1), 5: (4, 1)})
    variations.append({4: (3, 1), 5: (5, 1)})
    variations.append({3: (1, 1), 4: (2, 1), 5: (4, 1)})
    variations.append({3: (1, 1), 4: (3, 1), 5: (1, 6)})
    
    for var in variations:
        if time.time() > deadline:
            break
        
        try:
            n_var, edges_var = _luw_graph(k, g, rule_override=var)
            
            if n_var < best_n:
                # Validate
                if _validate_graph(n_var, edges_var, k, g, deadline=deadline):
                    best_n = n_var
                    best_edges = edges_var
        except Exception:
            continue
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
