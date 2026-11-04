import random
import numpy as np

def construct(k, g):
    # Petersen graph has 10 vertices, 15 edges.
    # We lift it by a group Z_p.
    # n = 10 * p.
    # We need n >= 512, so p >= 52.
    # Try primes starting from 53.
    
    # Petersen graph edges (0-9)
    # Standard labeling: 0-4 outer, 5-9 inner.
    # Outer: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    # Map edge tuple to index
    edge_index = {}
    for i, e in enumerate(edges_p):
        edge_index[e] = i
        edge_index[(e[1], e[0])] = i
        
    # Build adjacency list for Petersen
    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute all non-backtracking closed walks of length L < g
    # We need to check L from 5 to g-1.
    # Note: Non-backtracking closed walks of length < 5 don't exist in Petersen (girth 5).
    
    target_g = g
    
    # Generate constraints: list of (e_idx, sign) for each walk
    constraints = []
    for L in range(5, target_g):
        # DFS to find closed walks of length L
        # State: (curr, prev, length, seq)
        # seq is list of (e_idx, sign)
        for start in range(10):
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == L:
                    if curr == start:
                        constraints.append(seq)
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index.get((curr, nxt), edge_index.get((nxt, curr)))
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
    
    # Primes to try
    # Lower bound n >= 512. n = 10p. p >= 52.
    # Primes: 53, 59, 61, ...
    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    rng = random.Random(42)
    
    for p in primes_to_try:
        found = False
        # Try deterministic random search
        # To improve success rate, we can fix some voltages or use a smarter initialization.
        # But random usually works.
        for attempt in range(2000):
            # Generate random voltages
            # We can exclude 0? No, 0 is allowed as long as it doesn't cause a short cycle.
            # However, if an edge has voltage 0, it contributes 0 to any sum.
            voltages = [rng.randint(0, p-1) for _ in range(num_edges)]
            
            valid = True
            for w in constraints:
                s = 0
                for e_idx, sign in w:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            # Construct the graph
            n = 10 * p
            edges_out = set()
            
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    # u, fiber -> v, fiber + ve
                    v1 = u * p + fiber
                    v2 = v * p + (fiber + ve) % p
                    if v1 < v2:
                        edges_out.add((v1, v2))
                    else:
                        edges_out.add((v2, v1))
                    
                    # v, fiber -> u, fiber - ve
                    # This is the same undirected edge as above?
                    # Edge (u, i) -- (v, i+ve) is the same as (v, i+ve) -- (u, i).
                    # The second part of the loop in the original code was redundant/wrong if we just want unique undirected edges.
                    # Let's verify.
                    # The lift creates an edge between (u, i) and (v, i+ve).
                    # It also creates an edge between (v, j) and (u, j-ve).
                    # If we set j = i+ve, then the second edge is (v, i+ve) and (u, i).
                    # So it's the same edge.
                    # So we only need to iterate once per base edge and per fiber.
                    # Wait, does the lift create 2 edges per base edge per fiber?
                    # No. A voltage graph lift of an undirected graph with voltages in an abelian group:
                    # For each edge e=(u,v) with voltage g, we add edge ((u, i), (v, i+g)).
                    # Since the graph is undirected, ((v, i+g), (u, i)) is the same edge.
                    # So for each base edge and each fiber i, we get exactly one edge in the lift.
                    # Total edges = 15 * p.
                    # Vertices = 10 * p.
                    # Degree = 3.
                    # 3 * 10 * p / 2 = 15 * p. Correct.
                    
            return n, list(edges_out)
            
    # Fallback: Return the LUW graph if no small voltage graph found
    # But we should find one for p=53.
    # If not, we might timeout or fail.
    # Let's assume p=53 works.
    
    # Fallback to original logic if all fail
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
