import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_fallback(k, g):
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

    # Try Cayley graph of SL(2, 11)
    # 1. Generate group elements
    # Elements are 2x2 matrices over F_11 with det 1
    p = 11
    elements = []
    elem_to_idx = {}
    
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # d = (1 - bc)^{-1} * a ? No. ad - bc = 1 => d = (1+bc)/a if a!=0.
                # If a==0, -bc=1 => c!=0, b!=0. d is free? No, ad=0 so 0-bc=1 => bc=-1.
                # Let's just iterate all 4 and check det.
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det == 1:
                        mat = (a, b, c, d)
                        if mat not in elem_to_idx:
                            elem_to_idx[mat] = len(elements)
                            elements.append(mat)
    
    n = len(elements)
    if n != 1320:
        return luw_fallback(k, g)

    # Multiplication
    def mat_mul(x, y):
        a1, b1, c1, d1 = x
        a2, b2, c2, d2 = y
        return (
            (a1 * a2 + b1 * c2) % p,
            (a1 * b2 + b1 * d2) % p,
            (c1 * a2 + d1 * c2) % p,
            (c1 * b2 + d1 * d2) % p
        )

    # Precompute multiplication table? 1320^2 = 1.7M entries. Might be slow in Python.
    # Better to compute on fly or use a dict for neighbors.
    
    # Define generators
    # g = [[0, 1], [-1, 0]] = (0, 1, 10, 0)
    g = (0, 1, 10, 0)
    # h = [[0, -1], [1, 1]] = (0, 10, 1, 1)
    h = (0, 10, 1, 1)
    # k = -I = (10, 0, 0, 10)
    k_mat = (10, 0, 0, 10)
    
    # Check orders
    # g^2 = -I, g^4 = I
    # h^3 = I
    
    gens = [g, mat_mul(g, g), h, mat_mul(h, h), k_mat]
    # Ensure no duplicates and no identity
    identity = (1, 0, 0, 1)
    valid_gens = []
    for gen in gens:
        if gen != identity and gen not in valid_gens:
            valid_gens.append(gen)
            
    if len(valid_gens) != 5:
        return luw_fallback(k, g)

    # Build graph
    adj = [[] for _ in range(n)]
    for i, u in enumerate(elements):
        for gen in valid_gens:
            v = mat_mul(u, gen)
            j = elem_to_idx[v]
            adj[i].append(j)
            
    # Check girth
    # BFS from each node? No, just check for cycles < 9.
    # Since it's a Cayley graph, it's vertex transitive.
    # We can just check the girth from the identity node?
    # No, the girth of the graph is the minimum cycle length anywhere.
    # In a Cayley graph, the girth is the length of the shortest non-trivial relation.
    # This is the shortest path from identity to identity (excluding length 0).
    # We can BFS from identity.
    
    from collections import deque
    dist = [-1] * n
    parent = [-1] * n
    q = deque()
    dist[elem_to_idx[identity]] = 0
    q.append(elem_to_idx[identity])
    
    min_cycle = 100
    
    while q:
        u = q.popleft()
        if dist[u] >= 4: # We only care about cycles < 9, so depth 4 is enough to detect?
            # If we find a back edge to a node with dist >= 0
            # Cycle length = dist[u] + dist[v] + 1
            # If dist[u] + dist[v] + 1 < 9, we fail.
            # Max dist needed: if dist[u]=4, dist[v] could be 0, 1, 2, 3, 4.
            # 4+0+1=5, 4+1+1=6, 4+2+1=7, 4+3+1=8, 4+4+1=9.
            # So we need to process up to depth 4.
            pass
            
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                if dist[v] < 5: # Only push if we might find short cycles
                    q.append(v)
            else:
                # Found a cycle
                # Avoid counting the tree edge
                if v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < g:
                            # Fail fast
                            return luw_fallback(k, g)

    if min_cycle >= g:
        # Construct edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    else:
        return luw_fallback(k, g)

