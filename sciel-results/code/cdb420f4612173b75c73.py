"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    
    Strategy: Use voltage graph / group-lifting approach.
    For k=3, g=16: Lift the dodecahedron (3-regular, girth 5) using
    voltage assignments from finite groups to get larger graphs with
    girth multiplied by the group order (approximately).
    
    Also try: systematic search over circulant graphs and LUW with
    optimized parameters.
    """
    deadline = time.time() + 115.0
    
    if k != 3:
        return _luw_construct(k, g)
    
    best_n = None
    best_edges = None
    
    # Strategy 1: Voltage lift of dodecahedron
    # The dodecahedron has 20 vertices, 30 edges, girth 5.
    # Lifting by group G of order m gives a graph with 20*m vertices.
    # Girth is approximately 5 * (girth of G with respect to the voltage set).
    # For girth 16, we need the lifted girth >= 16.
    
    # Dodecahedron structure
    dodeca = _dodecahedron()
    dodeca_n, dodeca_edges, dodeca_adj = dodeca
    
    # Try various group orders for lifting
    # We need final girth >= 16, base girth is 5.
    # If we lift by group of order m, the girth can be up to 5*m
    # (if all short cycles in the lift come from base cycles).
    # So m >= ceil(16/5) = 4 would give girth up to 20.
    # But we need to check actual girth.
    
    # For the lift to work, we assign to each edge a group element (voltage).
    # The lifted graph has vertices (v, g) where v is a dodecahedron vertex
    # and g is a group element. Edge (u,v) with voltage w connects (u,g) to (v, g*w).
    
    # We'll try cyclic groups Z_m for various m, with different voltage assignments.
    
    for m in range(4, 20):
        if time.time() > deadline - 20:
            break
        
        # Try several voltage assignments
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Random voltage assignment from Z_m
            rng = random.Random(42 + attempt * 1000 + m)
            voltages = {}
            for u, v in dodeca_edges:
                voltages[(u, v)] = rng.randint(0, m - 1)
                voltages[(v, u)] = (-voltages[(u, v)]) % m
            
            # Build lifted graph
            lifted_n = dodeca_n * m
            lifted_edges = []
            for u, v in dodeca_edges:
                w = voltages[(u, v)]
                for g in range(m):
                    gu = u * m + g
                    gv = v * m + ((g + w) % m)
                    if gu < gv:
                        lifted_edges.append((gu, gv))
                    else:
                        lifted_edges.append((gv, gu))
            
            # Check girth
            girth = _check_girth_fast(lifted_n, lifted_edges, g)
            if girth >= g:
                if best_n is None or lifted_n < best_n:
                    best_n = lifted_n
                    best_edges = lifted_edges
                    if best_n <= 200:
                        return best_n, best_edges
    
    # Strategy 2: Try LUW with various parameters
    if time.time() < deadline - 10:
        for q in [3, 5, 7]:
            for d in range(5, 14):
                if time.time() > deadline - 10:
                    break
                try:
                    n, edges = _luw_construct_param(k, g, q, d)
                    if n is not None:
                        girth = _check_girth_fast(n, edges, g)
                        if girth >= g:
                            if best_n is None or n < best_n:
                                best_n = n
                                best_edges = edges
                except Exception:
                    pass
    
    # Strategy 3: Random 3-regular graphs (fallback)
    if best_n is None and time.time() < deadline - 5:
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            n = 512 + trial * 32
            edges = _random_3reg(n)
            if edges is None:
                continue
            girth = _check_girth_fast(n, edges, g)
            if girth >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                if best_n <= 512:
                    break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Final fallback: LUW with default params
    return _luw_construct(k, g)


def _dodecahedron():
    """Return the dodecahedron graph: (n, edges, adj)."""
    # Dodecahedron: 20 vertices, 30 edges, 3-regular, girth 5
    # Vertices: 2 outer pentagon (0-4), 5 upper (5-9), 5 lower (10-14), 2 inner pentagon (15-19)
    # Actually let me use the standard construction.
    # 
    # Standard dodecahedron vertices (from graph theory references):
    # Outer ring: 0,1,2,3,4 (pentagon)
    # Upper ring: 5,6,7,8,9 (pentagon, offset)
    # Lower ring: 10,11,12,13,14 (pentagon, offset)
    # Inner ring: 15,16,17,18,19 (pentagon)
    #
    # Edges:
    # Outer pentagon: (0,1),(1,2),(2,3),(3,4),(4,0)
    # Inner pentagon: (15,16),(16,17),(17,18),(18,19),(19,15)
    # Upper pentagon: (5,6),(6,7),(7,8),(8,9),(9,5)
    # Lower pentagon: (10,11),(11,12),(12,13),(13,14),(14,10)
    # Connections outer-upper: (0,5),(1,6),(2,7),(3,8),(4,9)
    # Connections upper-lower: (5,10),(6,11),(7,12),(8,13),(9,14)
    # Connections lower-inner: (10,15),(11,16),(12,17),(13,18),(14,19)
    
    n = 20
    edges = [
        # Outer pentagon
        (0,1),(1,2),(2,3),(3,4),(4,0),
        # Inner pentagon
        (15,16),(16,17),(17,18),(18,19),(19,15),
        # Upper pentagon
        (5,6),(6,7),(7,8),(8,9),(9,5),
        # Lower pentagon
        (10,11),(11,12),(12,13),(13,14),(14,10),
        # Outer to upper
        (0,5),(1,6),(2,7),(3,8),(4,9),
        # Upper to lower
        (5,10),(6,11),(7,12),(8,13),(9,14),
        # Lower to inner
        (10,15),(11,16),(12,17),(13,18),(14,19),
    ]
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    return n, edges, adj


def _check_girth_fast(n, edges, target_girth):
    """
    Check if graph has girth >= target_girth using non-backtracking walks.
    Returns the actual girth if < target_girth, else target_girth.
    Uses BFS with non-backtracking constraint.
    """
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = target_girth
    
    # For each edge (u,v), find shortest cycle containing that edge.
    # A cycle of length L containing edge (u,v) means a path of length L-1
    # from u to v that doesn't use the edge (u,v) and doesn't backtrack.
    
    # More efficient: for each directed edge (u,v), do BFS from v avoiding u,
    # and find shortest path back to u. Cycle length = path_length + 1.
    
    for u, v in edges:
        # BFS from v, can't go to u initially (non-backtracking)
        # Track: dist[node], parent[node]
        dist = [-1] * n
        dist[v] = 0
        parent = [-1] * n
        queue = deque([v])
        
        found = False
        while queue:
            node = queue.popleft()
            if dist[node] >= target_girth - 1:
                break
            for nb in adj[node]:
                if nb == parent[node]:
                    continue  # Non-backtracking
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    if nb == u:
                        # Found cycle of length dist[node] + 1 + 1 = dist[node] + 2
                        cycle_len = dist[node] + 2
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        found = True
                        break
                    queue.append(nb)
                elif nb == u and dist[node] + 2 < min_girth:
                    # Found shorter cycle via this edge
                    cycle_len = dist[node] + 2
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    found = True
                    break
            if found:
                break
    
    return min_girth


def _luw_construct(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction."""
    return _luw_construct_param(k, g, k, None)


def _luw_construct_param(k, g, q, d):
    """LUW construction with specified parameters."""
    if d is None:
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


def _random_3reg(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
    
    stubs = []
    for v in range(n):
        stubs.extend([v, v, v])
    
    attempts = 0
    while attempts < 50:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
        
        if valid and len(edges) == 3 * n // 2:
            return list(edges)
        attempts += 1
    
    return None
# EVOLVE-BLOCK-END
