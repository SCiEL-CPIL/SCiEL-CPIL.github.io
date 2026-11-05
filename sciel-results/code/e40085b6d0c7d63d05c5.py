"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
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

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Selects the optimal d to minimize vertices while satisfying girth >= g.
    Girth for odd d is d+5.
    Girth for even d is 2d+2.
    Returns (n, edges).
    """
    q = k
    
    # Determine optimal d
    best_d = None
    best_n = float('inf')
    
    # Check d from 2 up to a reasonable limit (e.g., 10)
    for d in range(2, 11):
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = 2 * d + 2
            
        if girth >= g:
            n = 2 * (q ** d)
            if n < best_n:
                best_n = n
                best_d = d
            # Since n grows exponentially, once we find a valid d for a parity,
            # larger d of the same parity will be worse. 
            # But we must check the other parity.
            # We can break if n is already larger than the best known for the other parity?
            # Simpler: just check all d up to 10, it's fast.
            
    if best_d is None:
        # Fallback to a large d if something went wrong
        best_d = 5 if g <= 10 else g - 5
        if best_d % 2 == 0 and (2*best_d+2) < g:
            best_d += 1
        if best_d % 2 == 1 and (best_d+5) < g:
            best_d += 1
            
    d = best_d
    
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
    
    # 1. Try LUW Graph with optimal d
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. (Optional) If we have time and k,g are small, we could try to search for smaller,
    # but LUW with optimal d is already a strong baseline.
    # For k=4, g=10, LUW with d=4 gives n=512.
    # The Foster graph is n=320. 
    # Can we find 320? 
    # The gap between 512 and 320 is significant for the score (512/320 = 1.6x improvement).
    # However, constructing the Foster graph requires specific knowledge.
    # Given the constraints, the LUW improvement is the safest and most robust "jump".
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
