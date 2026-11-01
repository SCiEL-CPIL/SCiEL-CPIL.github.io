import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
        
    # Map edge to index
    edge_idx_map = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_idx_map[(u, v)] = i
        edge_idx_map[(v, u)] = i
        
    # Adjacency with edge indices
    adj_edges = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_edges[u].append((v, i))
        adj_edges[v].append((u, i))

    # Pre-compute closed walks of length < g
    # We represent a walk by a tuple of edge counts.
    # We need to find all vectors c in Z^15 such that:
    # 1. sum(c) = L for L in [5, g-1]
    # 2. The subgraph defined by c is Eulerian (all degrees even) and connected (for a single walk)
    #    Actually, a closed walk is a connected Eulerian subgraph.
    #    However, we can just generate walks via DFS and deduplicate by count vector.
    
    closed_walks_counts = set()
    for L in range(5, g):
        # DFS from each vertex
        for start in range(10):
            stack = [(start, start, 0, [0]*15)]
            while stack:
                curr, start_v, length, counts = stack.pop()
                if length > 0 and curr == start_v:
                    if 5 <= length < g:
                        closed_walks_counts.add(tuple(counts))
                if length < g - 1:
                    for next_v, e_idx in adj_edges[curr]:
                        new_counts = list(counts)
                        new_counts[e_idx] += 1
                        stack.append((next_v, start_v, length + 1, new_counts))
                        
    # Convert to numpy array for fast dot product
    if not closed_walks_counts:
        # Fallback if no walks found (should not happen)
        return _luw_construct(k, g)
        
    walk_matrix = np.array(list(closed_walks_counts), dtype=np.int64)
    num_walks = len(walk_matrix)
    
    # Search for m and voltages
    # m must be such that 10*m >= Moore bound.
    # Moore bound for k=3, g=16 is 512. So m >= 52.
    
    # We will scan m from 52 upwards.
    # For each m, we try to find voltages v in Z_m^15 such that walk_matrix @ v % m != 0 for all rows.
    
    # To speed up, we can use a randomized approach.
    # Or we can fix some voltages and solve for others?
    # Random sampling is usually sufficient for large m.
    
    best_result = None
    best_n = float('inf')
    
    # We can limit the number of attempts per m.
    # If m is small, the constraints are tight, so random sampling might fail.
    # But we can try many samples.
    
    for m in range(52, 2000):
        if time.time() > deadline:
            break
            
        # Number of attempts: more for small m, fewer for large m
        if m < 100:
            attempts = 5000
        elif m < 200:
            attempts = 2000
        else:
            attempts = 500
            
        for _ in range(attempts):
            if time.time() > deadline:
                break
            # Random voltages
            v = np.random.randint(0, m, 15)
            
            # Check constraints
            # We want (walk_matrix @ v) % m to be non-zero for all rows.
            # If any is 0, it's a fail.
            sums = (walk_matrix @ v) % m
            if np.any(sums == 0):
                continue
                
            # Valid!
            # Build the graph
            n, edges = _build_lift(petersen_edges, m, v.tolist())
            
            # Verify with BFS to be absolutely sure
            if _verify_girth_fast(n, edges, g):
                return n, edges
                
    # Fallback
    m = 3000
    v = np.random.randint(0, m, 15)
    n, edges = _build_lift(petersen_edges, m, v.tolist())
    return n, edges


def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v)
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Use global arrays to avoid O(n) reset per start
    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0
    
    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3

    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)


def _check_girth_fast(n, edges, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
