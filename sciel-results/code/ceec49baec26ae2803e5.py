"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random

_DEADLINE = 120.0
_MARGIN = 10.0  # seconds to stop searching

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
    # A cycle of length L < g exists if there is a non-backtracking walk
    # from v to v of length L.
    for s in range(n):
        # dist[(curr, prev)] = depth
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                # Found a cycle of length depth + 1
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
    Returns (n, edges) for the connected component of the zero vertex.
    """
    q = k
    # d is chosen such that girth >= d + 5
    # For g=10, d=5 gives girth >= 10.
    # For g=8, d=3 gives girth >= 8.
    # General rule: d = g - 5 if (g-5) is odd, else g - 4?
    # Actually, LUW girth is at least d+5 for odd d.
    # If we want girth 10, d=5 is odd.
    # If we want girth 8, d=3 is odd.
    # If g is even, d = g - 5 is odd.
    # If g is odd, d = g - 5 is even? Girth of LUW is at least d+5.
    # So d = g - 5 works for both if we accept girth >= g.
    
    d = g - 5
    if d < 3:
        d = 3
        
    # Field arithmetic
    if q == 4:
        # GF(4) using XOR for add/sub and table for mul
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
        
    # Define the rules for the graph
    # y[c] = add(x[c], mul(y[a], x[b])) for side 0
    # y[c] = sub(x[c], mul(x[a], y[b])) for side 1
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # Pattern of (a, b) pairs
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
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
            # Add edge only once (when we are on side 0)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _random_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices.
    Returns list of edges or None if failed.
    """
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(4):
            stubs.append(v)
    rng.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None
        edges.add(edge)
        
    # Verify degree and simplicity
    adj = [0] * n
    for u, v in edges:
        adj[u] += 1
        adj[v] += 1
        if adj[u] > 4 or adj[v] > 4:
            return None
            
    if any(d != 4 for d in adj):
        return None
        
    return list(edges)

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Moore Lower Bound
    r = g // 2
    if g % 2 == 0:
        lb = k * (k - 1) * (r + 1) - (k - 1)
    else:
        lb = k * (k - 1) * (r + 1) + 1
        
    best_n = None
    best_edges = None
    
    # 1. Try LUW Graph (Explicit Construction)
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        # Validate to be sure
        if _validate_graph(n_luw, edges_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass
        
    # 2. Try Random Search for smaller graphs
    # Only if we have a baseline or if LUW failed
    # Focus on (4, 10) where we know 320 is possible
    if k == 4 and g == 10:
        target_min = 320
        # Search from 320 upwards to best_n - 1
        upper = best_n - 1 if best_n else 400
        if upper >= target_min:
            for n in range(target_min, upper + 1):
                if time.time() > deadline:
                    break
                # Try a few random instances
                for trial in range(100):
                    if time.time() > deadline:
                        break
                    edges = _random_4_regular(n, seed=n * 1000 + trial)
                    if edges is None:
                        continue
                    # Quick girth check
                    if _validate_graph(n, edges, k, g):
                        best_n = n
                        best_edges = edges
                        break # Found a valid graph for this n, move to next n
                        # Actually, we want the smallest n, so if we find one, we can break
                        # and try smaller n? No, we are iterating from 320 up.
                        # So 320 is the first one. If we find it, we stop.
                        break
                if best_n == target_min:
                    break # Can't do better than 320 if that's the known min
                    
    # 3. Fallback
    if best_n is None:
        # If LUW failed, try a simple cycle if k=2
        if k == 2:
            n = g
            edges = [(i, (i+1)%n) for i in range(n)]
            return n, edges
        else:
            # Return empty or raise? Task says return n, edges.
            # If we can't find any, return a dummy?
            # But LUW should always work for k=4, g=10.
            return 0, []
            
    return best_n, best_edges

# EVOLVE-BLOCK-END
