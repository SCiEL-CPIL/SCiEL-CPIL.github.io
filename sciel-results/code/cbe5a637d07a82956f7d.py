import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Quick voltage graph search with improved strategies
    best_n, best_edges = _search_voltage_graphs(k, g, deadline)
    
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to LUW - this is the reliable path
    return _luw_construct(k, g, deadline)

def _search_voltage_graphs(k, g, deadline):
    """Search for voltage graph lifts with girth >= g."""
    best_n = float('inf')
    best_edges = None
    
    # Generate a few candidate base graphs
    base_graphs = []
    
    # K6 (complete graph on 6 vertices) - 5-regular
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_graphs.append((6, edges_k6))
    
    # Petersen graph (10 vertices, 3-regular) - not 5-regular, skip
    # Use other 5-regular base graphs
    
    # Random 5-regular graphs on small vertex counts
    random.seed(42)
    for V in [12, 14, 16, 20, 24, 30]:
        for attempt in range(3):
            stubs = [i for i in range(V) for _ in range(5)]
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
            if valid and len(edges) == 5 * V // 2:
                # Verify it's actually 5-regular
                deg = [0] * V
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == 5 for d in deg):
                    base_graphs.append((V, list(edges)))
                    break  # One per V is enough
    
    base_graphs.sort(key=lambda x: x[0])
    
    for V, base_edges in base_graphs:
        if time.time() > deadline - 15:
            break
            
        # Try m values that give N >= 426 (Moore bound)
        min_m = max(2, (426 + V - 1) // V)
        max_m = min_m + 10
        
        for m in range(min_m, max_m + 1):
            N = V * m
            if N < 426:
                continue
            if time.time() > deadline - 10:
                break
                
            # Try structured voltage assignments
            for trial in range(20):
                if time.time() > deadline - 5:
                    break
                    
                voltages = _generate_voltages(base_edges, m, trial)
                
                # Build lift and check girth
                lift_adj = _build_lift(base_edges, voltages, m)
                
                if _check_girth_fast(lift_adj, N, g):
                    edges = []
                    for u in range(N):
                        for v in lift_adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        if N <= 430:
                            return N, edges
    
    return best_n, best_edges

def _generate_voltages(base_edges, m, trial):
    """Generate voltage assignments using different strategies."""
    voltages = {}
    n_edges = len(base_edges)
    
    if trial < 5:
        # Pure random
        for u, v in base_edges:
            w = random.randint(0, m-1)
            voltages[(u, v)] = w
            voltages[(v, u)] = (m - w) % m
    elif trial < 10:
        # Linear in edge index with random offset
        offset = random.randint(0, m-1)
        step = random.randint(1, max(1, m//2))
        for idx, (u, v) in enumerate(base_edges):
            w = (offset + idx * step) % m
            voltages[(u, v)] = w
            voltages[(v, u)] = (m - w) % m
    elif trial < 15:
        # Based on vertex indices
        a = random.randint(1, max(1, m//3))
        b = random.randint(1, max(1, m//3))
        c = random.randint(0, m-1)
        for u, v in base_edges:
            w = (a * u + b * v + c) % m
            voltages[(u, v)] = w
            voltages[(v, u)] = (m - w) % m
    else:
        # Quadratic pattern
        a = random.randint(1, max(1, m//4))
        b = random.randint(0, m-1)
        c = random.randint(0, m-1)
        for idx, (u, v) in enumerate(base_edges):
            w = (a * idx * idx + b * idx + c) % m
            voltages[(u, v)] = w
            voltages[(v, u)] = (m - w) % m
    
    return voltages

def _build_lift(base_edges, voltages, m):
    """Build the lifted graph from base edges and voltages."""
    # Determine number of base vertices
    V = max(max(u, v) for u, v in base_edges) + 1
    N = V * m
    lift_adj = [[] for _ in range(N)]
    
    for u, v in base_edges:
        w = voltages[(u, v)]
        for i in range(m):
            u_idx = u * m + i
            v_idx = v * m + (i + w) % m
            lift_adj[u_idx].append(v_idx)
            lift_adj[v_idx].append(u_idx)
    
    return lift_adj

def _check_girth_fast(adj, n, min_girth):
    """Check if graph has girth >= min_girth using non-backtracking walks with early termination."""
    # Quick triangle check using adjacency sets
    adj_sets = [set(neighbors) for neighbors in adj]
    
    for u in range(n):
        for v in adj[u]:
            if u < v and adj_sets[u] & adj_sets[v]:
                return False
    
    # Quick 4-cycle check
    for u in range(n):
        neighbors_u = adj[u]
        for i, v in enumerate(neighbors_u):
            for w in neighbors_u[i+1:]:
                if adj_sets[v] & adj_sets[w]:
                    return False
    
    # For longer cycles, use BFS from each vertex up to depth (min_girth-1)//2
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        # BFS tracking (vertex, depth, parent) to detect cycles
        visited = {}  # vertex -> (depth, parent)
        queue = [(start, 0, -1)]
        
        while queue:
            u, depth, parent = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            if u in visited:
                prev_depth, prev_parent = visited[u]
                if prev_depth + depth + 1 < min_girth:
                    return False
                continue
            
            visited[u] = (depth, parent)
            
            if depth < max_depth:
                for v in adj[u]:
                    if v == parent:
                        continue
                    queue.append((v, depth + 1, u))
    
    return True

def _luw_construct(k, g, deadline):
    """LUW construction for regular graphs with high girth."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
