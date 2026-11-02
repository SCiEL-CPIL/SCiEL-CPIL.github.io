"""Small 4-regular graph of girth 10 (cage problem n(4,10))."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # For k=4, g=10, we know the optimal is 320.
    # We will try to find a random 4-regular graph with girth >= 10 for n < 320.
    # If that fails, we will try n up to 2000.
    # If that fails, we fall back to the LUW construction (which gives 2048).
    
    if k == 4 and g == 10:
        # Try random search for smaller n
        # The Moore bound is 161.
        # Try n in a range likely to contain solutions.
        # Random regular graphs have girth ~ log_k(n).
        # For n=200, girth is often > 10.
        
        # We will iterate n from 162 to 1000 (step 2, must be even for 4-regular? No, n*k must be even. 4n is always even).
        # But girth 10 implies no odd cycles < 10.
        
        # Optimization: Check girth efficiently.
        
        # We'll try a few n values.
        target_ns = [162, 164, 166, 168, 170, 180, 200, 220, 240, 260, 280, 300, 320, 350, 400, 500, 600, 800, 1000]
        
        for n in target_ns:
            if time.time() > deadline - 5:
                break
            
            # Try to find a valid graph for this n
            result = _random_search(n, k, g, deadline)
            if result is not None:
                return result
        
        # If random search failed, use LUW
        return _luw_construct(k, g)
    else:
        # Fallback for other parameters
        return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def _random_search(n, k, g, deadline):
    """Try to find a random k-regular graph with girth >= g on n vertices."""
    # Number of attempts depends on time
    max_attempts = 500
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            return None
        
        edges = _gen_random_4reg(n)
        if edges is None:
            continue
        
        if _has_girth_at_least(edges, n, g):
            return n, edges
            
    return None


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    # Use configuration model
    for attempt in range(20):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None


def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Optimization: We only need to check cycles of length < g.
    # A cycle of length L is found if a vertex v is reached at distance d1 and d2 such that d1 + d2 + 1 = L?
    # No, standard BFS detects cycles.
    # If we do BFS from start, and encounter a visited node v (not parent),
    # the cycle length is dist[u] + dist[v] + 1.
    # We need to ensure this is >= g.
    
    max_depth = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
# EVOLVE-BLOCK-END
