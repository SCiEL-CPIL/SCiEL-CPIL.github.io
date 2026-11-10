import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (evaluator kills at 150s)
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9) is approximately 428
    moore_bound = 2 + k * (1 + k * (1 + k * (1 + k)))  # This is the Moore bound formula
    # Actually for regular graphs: n >= 2 + k * sum_{i=0}^{floor((g-2)/2)-1} k^i
    # For g=9: floor((9-2)/2) = 3, so n >= 2 + 5*(1 + 5 + 25) = 2 + 5*31 = 157
    # Wait, let me recalculate. Moore bound for (k,g):
    # n >= 2 + k * (k-1) * (k-1)^{floor((g-2)/2) - 1} / ((k-1) - 1) ... this is getting complicated
    # For k=5, g=9: n >= 2 + 5 * (1 + 4 + 16) = 2 + 5*21 = 107? No.
    # The Moore bound is: n >= 2 + k * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    # For g=9: floor((9-2)/2) = 3, so sum from i=0 to 2: 1 + 4 + 16 = 21
    # n >= 2 + 5*21 = 107. But the actual lower bound is 428, so there's a better bound.
    
    # Let's just use 428 as the lower bound from the task description
    
    # Strategy 1: Try voltage graph lifts with various base graphs
    # Strategy 2: Fall back to LUW construction
    
    # First, try to find a small graph via voltage graphs
    candidates = []
    for v_b in range(k + 1, 30):
        if (k * v_b) % 2 != 0:
            continue
        for m in range(30, 300):
            n = v_b * m
            if n < 428:
                continue
            if n > 1200:
                break
            candidates.append((v_b, m))
    
    random.shuffle(candidates)
    
    for v_b, m in candidates:
        if time.time() > deadline:
            break
            
        graph = _random_regular_graph(v_b, k)
        if graph is None:
            continue
            
        edges_b = graph
        n_edges = len(edges_b)
        
        # Precompute all closed walks in base graph up to length g-1
        # For each closed walk, we need sum of voltages != 0 mod m
        # This is a system of linear constraints
        
        # Try random voltage assignments
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            
            derived_edges = _construct_derived_graph_correct(edges_b, voltages, m)
            n = v_b * m
            
            if _check_girth_fast(derived_edges, n, k, g):
                return n, derived_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _random_regular_graph(v, k):
    """Generate a random k-regular simple graph on v vertices."""
    if (v * k) % 2 != 0:
        return None
    if k >= v:
        return None
    if k == v - 1:
        # Complete graph
        return [(i, j) for i in range(v) for j in range(i+1, v)]
    
    # Configuration model with retries
    for _ in range(100):
        stubs = []
        for i in range(v):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v2 = stubs[i], stubs[i+1]
            if u == v2:
                valid = False
                break
            edge = (min(u, v2), max(u, v2))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

def _construct_derived_graph_correct(base_edges, voltages, m):
    """Construct the derived graph from a voltage graph.
    
    For each undirected edge (u, v) in the base graph with voltage w:
    - (u, a) is connected to (v, (a + w) % m)
    
    Since the graph is undirected, we only add each edge once.
    """
    v_b = max([max(e) for e in base_edges]) + 1 if base_edges else 1
    derived_edges = []
    
    for idx, (u, v) in enumerate(base_edges):
        w = voltages[idx]
        for a in range(m):
            u_idx = u * m + a
            v_idx = v * m + (a + w) % m
            derived_edges.append((u_idx, v_idx))
    
    return derived_edges

def _check_girth_fast(edges, n, k, g):
    """Check if the graph has girth >= g using BFS from each vertex.
    
    For efficiency, we only need to check up to depth (g-1)//2.
    If we find a cycle of length < g, return False.
    """
    if not edges:
        return True
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    max_depth = (g - 1) // 2
    
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[v] != u and parent[u] != v:
                    # Found a cycle: u - v where both are visited
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
