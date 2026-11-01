import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Try voltage graph construction first
    best_n, best_edges = _voltage_search(deadline)
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _voltage_search(deadline):
    """Search for voltage graph lifts with good girth properties."""
    best_n = float('inf')
    best_edges = None
    
    # Base graphs: try K6, K5,5, and some structured ones
    base_candidates = []
    
    # K6 (6 vertices, 15 edges)
    edges_k6 = [(i, j) for i in range(6) for j in range(i+1, 6)]
    base_candidates.append((6, edges_k6))
    
    # K5,5 (10 vertices, 25 edges)  
    edges_k55 = [(i, j+5) for i in range(5) for j in range(5)]
    base_candidates.append((10, edges_k55))
    
    # Petersen-like: 10-vertex 3-regular won't work, need 5-regular
    # Try 12-vertex 5-regular: complement of perfect matching in K12
    # K12 has degree 11, remove perfect matching (6 edges) to get degree 10, not 5
    # Actually, let's just use random 5-regular graphs
    
    # Generate random 5-regular base graphs
    for V in [12, 14, 16, 18, 20]:
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            base = _random_5_regular(V)
            if base is not None:
                base_candidates.append((V, base))
    
    # For each base, try voltage assignments
    for V, base_edges in base_candidates:
        if time.time() > deadline - 15:
            break
        
        # Target: N = V * m >= 428 (Moore bound)
        # Try m values that give N close to 428-600
        target_m = max(2, 428 // V)
        
        for m in range(target_m, target_m + 10):
            N = V * m
            if N < 428:
                continue
            if N > 1200:
                break
            
            # Try several random voltage assignments
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                voltages = {}
                for u, v in base_edges:
                    w = random.randint(0, m-1)
                    voltages[(u, v)] = w
                    voltages[(v, u)] = (-w) % m
                
                # Build lift
                adj = [[] for _ in range(N)]
                for u, v in base_edges:
                    w = voltages[(u, v)]
                    for i in range(m):
                        ui = u * m + i
                        vi = v * m + (i + w) % m
                        adj[ui].append(vi)
                        adj[vi].append(ui)
                
                # Quick girth check using BFS
                if _has_girth_at_least(adj, N, 9):
                    edges = []
                    for u in range(N):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                        if N <= 450:
                            return best_n, best_edges
    
    return best_n, best_edges

def _random_5_regular(V):
    """Generate a random 5-regular graph on V vertices using configuration model."""
    if V < 6 or V * 5 % 2 != 0:
        return None
    
    for _ in range(20):
        stubs = [i for i in range(V) for _ in range(5)]
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if valid and len(edges) == V * 5 // 2:
            # Verify degree
            deg = [0] * V
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == 5 for d in deg):
                return list(edges)
    
    return None

def _has_girth_at_least(adj, n, min_girth):
    """Check if graph has girth at least min_girth using non-backtracking walks."""
    # For each vertex, do BFS tracking the previous vertex to avoid backtracking
    for start in range(n):
        # dist[v] = shortest non-backtracking path length from start to v
        # We track (current, prev) states
        visited = set()
        queue = [(start, -1, 0)]  # (node, prev, dist)
        visited.add((start, -1))
        
        while queue:
            node, prev, dist = queue.pop(0)
            
            if dist >= min_girth:
                return False  # Found a cycle of length < min_girth? No, this is just long paths
            
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue  # Don't backtrack
                state = (neighbor, node)
                if state not in visited:
                    visited.add(state)
                    if neighbor == start and dist + 1 >= 3:
                        # Found a cycle
                        cycle_len = dist + 1
                        if cycle_len < min_girth:
                            return False
                    queue.append((neighbor, node, dist + 1))
    
    return True

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
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
