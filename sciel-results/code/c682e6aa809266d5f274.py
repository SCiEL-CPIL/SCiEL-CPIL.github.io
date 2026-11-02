import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Base graph: K_{4,4}
    # Vertices 0,1,2,3 (Side 0) and 4,5,6,7 (Side 1)
    # Edges are all (i, j) with i in 0..3, j in 4..7
    
    # Search for m such that 8*m is minimized
    # Start from m=31 (N=248) up to m=50 (N=400)
    # Known best 320 (m=40)
    
    for m in range(31, 60):
        if time.time() > deadline - 5:
            break
        n_vertices = 8 * m
        
        # Local search for voltage assignment
        # Initialize random
        voltages = {}
        for a in range(4):
            for b in range(4):
                voltages[(a, b)] = random.randint(0, m - 1)
        
        # Build edges and check
        edges = build_edges(m, voltages)
        min_g = check_girth(n_vertices, edges, g)
        
        # Local search loop
        max_iterations = 5000
        for it in range(max_iterations):
            if time.time() > deadline - 2:
                break
            if min_g >= g:
                return n_vertices, edges
            
            # Pick a random edge in base graph
            a = random.randint(0, 3)
            b = random.randint(0, 3)
            old_v = voltages[(a, b)]
            new_v = random.randint(0, m - 1)
            if new_v == old_v:
                continue
            
            voltages[(a, b)] = new_v
            edges = build_edges(m, voltages)
            new_min_g = check_girth(n_vertices, edges, g)
            
            if new_min_g < min_g:
                min_g = new_min_g
            else:
                # Revert
                voltages[(a, b)] = old_v
                edges = build_edges(m, voltages)
                # min_g remains the old best
        
    # Fallback: LUW construction from seed
    return luw_construct(k, g)

def build_edges(m, voltages):
    edges = []
    for a in range(4):
        for b in range(4):
            v = voltages[(a, b)]
            base_a = a
            base_b = 4 + b
            for i in range(m):
                u_idx = base_a * m + i
                j_idx = (i + v) % m
                v_idx = base_b * m + j_idx
                edges.append((u_idx, v_idx))
    return edges

def check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_g = float('inf')
    # We only care if girth < g
    # So we can stop early if we find cycle < g
    for i in range(n):
        dist = [-1] * n
        dist[i] = 0
        queue = [i]
        # Max depth needed: if cycle length < g, 
        # e.g. g=10, cycle 9. Depth 4 is enough.
        max_depth = g // 2 
        for u in queue:
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] < dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_g:
                        min_g = cycle_len
                    if min_g < g:
                        return min_g
    return min_g

def luw_construct(k, g):
    # Copy of seed logic
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
