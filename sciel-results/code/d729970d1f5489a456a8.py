import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Handle the specific case k=5, g=9 with edge-switching strategy
    if k == 5 and g == 9:
        result = _edge_switching_k55()
        if result is not None:
            return result
        # Fallback to LUW if edge switching fails or times out
        return _luw_construct(k, g)
    
    # General fallback to LUW
    return _luw_construct(k, g)

def _edge_switching_k55():
    """
    Constructs a 5-regular graph of girth 9 using probabilistic edge switching.
    Starts from a random 5-regular graph and accepts switches that increase girth
    or maintain it at 9. Uses non-backtracking walks for local girth verification.
    """
    start_time = time.time()
    deadline = start_time + 108
    
    # Moore bound for k=5, g=9: 2 + 5*sum_{i=0}^{3} 4^i = 2 + 5*(1+4+16+64) = 2 + 5*85 = 427
    # Start near Moore bound and increase n if needed
    moore_bound = 428
    max_n = 1200  # Upper bound for search
    
    # Try different sizes starting from Moore bound
    for n in range(moore_bound, max_n + 1, 4):  # n must be even for 5-regular
        if time.time() > deadline - 5:
            break
        
        # Try to find a 5-regular graph of girth 9 on n vertices
        result = _switch_to_girth9(n, deadline)
        if result is not None:
            return result
    
    return None

def _switch_to_girth9(n, deadline):
    """
    Try to construct a 5-regular graph of girth 9 on n vertices using edge switching.
    Returns (n, edges) if successful, None otherwise.
    """
    # Generate initial random 5-regular graph
    adj = _random_5_regular(n)
    if adj is None:
        return None
    
    # Track girth (minimum cycle length)
    current_girth = _compute_girth(adj, n)
    
    # If already girth >= 9, we're done
    if current_girth >= 9:
        edges = _adj_to_edges(adj)
        return n, edges
    
    # Edge switching: repeatedly pick two non-adjacent edges and swap endpoints
    # to improve girth
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline - 2:
        iteration += 1
        
        # Pick two random non-adjacent edges (u,v) and (x,y) where no vertices overlap
        # and the new edges (u,x) and (v,y) don't already exist
        for _ in range(20):  # Try up to 20 times to find valid switch
            u, v = _random_edge(adj)
            x, y = _random_edge(adj)
            
            # Check no vertex overlap
            if len({u, v, x, y}) < 4:
                continue
            
            # Check new edges don't exist
            if x in adj[u] or y in adj[v]:
                continue
            
            # Perform the switch: remove (u,v) and (x,y), add (u,x) and (v,y)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Compute new girth
            new_girth = _compute_girth(adj, n)
            
            if new_girth >= 9:
                # Success!
                edges = _adj_to_edges(adj)
                return n, edges
            elif new_girth > current_girth:
                # Accept: girth improved
                current_girth = new_girth
            else:
                # Reject: revert the switch
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
        
        # Periodic check
        if iteration % 1000 == 0 and current_girth >= 9:
            edges = _adj_to_edges(adj)
            return n, edges
    
    return None

def _random_5_regular(n):
    """
    Generate a random 5-regular graph on n vertices.
    Returns adjacency list or None if impossible.
    """
    if n % 2 != 0 or n < 6:
        return None
    
    # Use configuration model with rejection
    for attempt in range(100):
        # Create degree sequence: each vertex has degree 5
        half_edges = [i for i in range(n) for _ in range(5)]
        random.shuffle(half_edges)
        
        # Pair up to form edges
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            # Verify all degrees are exactly 5
            if all(len(adj[i]) == 5 for i in range(n)):
                return adj
    
    return None

def _random_edge(adj):
    """Pick a random edge from the adjacency list."""
    n = len(adj)
    while True:
        u = random.randint(0, n - 1)
        if adj[u]:
            v = random.choice(adj[u])
            return (u, v)

def _compute_girth(adj, n):
    """
    Compute the girth (minimum cycle length) of a graph using BFS.
    Returns girth or a large number if no cycle found.
    """
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node] and parent[neighbor] != node:
                    # Found a cycle
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    # Early termination if we found a very short cycle
                    if min_cycle <= 3:
                        return min_cycle
        
        # Early termination
        if min_cycle <= 3:
            return min_cycle
    
    return min_cycle if min_cycle != float('inf') else n + 1

def _adj_to_edges(adj):
    """Convert adjacency list to edge list (each edge once)."""
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

# _voltage_lift_k55 removed - replaced by _edge_switching_k55

# _verify_girth removed - replaced by _compute_girth

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    # For k=5, g=9: use q=5, d=4 (girth 9 = 2*4+1)
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
